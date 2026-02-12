//! WebSocket server for Soprano TTS
//!
//! Provides real-time text-to-speech streaming via WebSocket connections.
//! Uses Axum for HTTP/WebSocket handling with proper async/await patterns.

use std::{collections::HashMap, future::Future, sync::Arc};

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{
    future::{BoxFuture, FusedFuture},
    stream::FuturesUnordered,
    FutureExt, SinkExt, StreamExt,
};
use serde::Serialize;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot, RwLock, Semaphore},
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    chunker::{Chunk, ChunkerInput},
    config::{EngineId, StreamConfig},
    protocol::{ClientMessage, ServerMessage},
    tts::{TtsEngine, TtsRequest, TtsResponse},
};

/// Application state shared across all connections
#[derive(Clone)]
pub struct AppState {
    /// TTS engine for synthesis
    pub tts: Arc<dyn TtsEngine>,
    /// Engine backing this server instance
    pub engine: EngineId,
    /// Default streaming configuration
    pub default_config: Arc<RwLock<StreamConfig>>,
    /// Semaphore to limit concurrent TTS requests
    pub tts_inflight: Arc<Semaphore>,
    /// Maximum number of concurrent TTS requests
    pub tts_inflight_limit: usize,
    /// Whether to include text in audio frames
    pub include_text: bool,
}

/// Result from a TTS job
struct TtsJobResult {
    id: u64,
    result: Result<TtsResponse, String>,
}

/// Convert f32 PCM samples to i16 bytes
fn pcm_f32_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&i16_sample.to_le_bytes());
    }
    bytes
}

impl AppState {
    /// Create a new application state
    pub fn new(
        tts: Arc<dyn TtsEngine>,
        config: StreamConfig,
        tts_inflight: usize,
        include_text: bool,
    ) -> Self {
        Self::new_with_engine(tts, EngineId::Soprano, config, tts_inflight, include_text)
    }

    /// Create a new application state with explicit engine routing.
    pub fn new_with_engine(
        tts: Arc<dyn TtsEngine>,
        engine: EngineId,
        config: StreamConfig,
        tts_inflight: usize,
        include_text: bool,
    ) -> Self {
        let limit = tts_inflight.max(1);
        Self {
            tts,
            engine,
            default_config: Arc::new(RwLock::new(config)),
            tts_inflight: Arc::new(Semaphore::new(limit)),
            tts_inflight_limit: limit,
            include_text,
        }
    }
}

/// Create the Axum router with all routes
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/stream", get(ws_handler))
        .with_state(state)
}

/// Start the server and serve requests indefinitely
pub async fn serve(
    listener: TcpListener,
    state: AppState,
) -> Result<(), Box<dyn std::error::Error>> {
    serve_with_shutdown(listener, state, std::future::pending()).await
}

/// Start the server with graceful shutdown support
pub async fn serve_with_shutdown<F>(
    listener: TcpListener,
    state: AppState,
    shutdown: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Future<Output = ()> + Send + 'static,
{
    let app = router(state);
    info!("starting axum server");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    info!("server shut down gracefully");
    Ok(())
}

/// Health check endpoint
async fn healthz() -> &'static str {
    "ok"
}

/// WebSocket upgrade handler
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Main WebSocket connection handler
async fn handle_socket(socket: WebSocket, state: AppState) {
    let session_id = Uuid::new_v4().to_string();
    info!(session_id = %session_id, "new websocket connection");

    // Clone config for this connection
    let config = Arc::new(RwLock::new(state.default_config.read().await.clone()));
    let selected_engine = Arc::new(RwLock::new(state.engine));

    // Channel for outgoing WebSocket messages
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(16);

    // Channels for chunker communication
    let (chunk_in_tx, chunk_in_rx) = mpsc::channel::<ChunkerInput>(32);
    let (chunk_out_tx, mut chunk_out_rx) = mpsc::channel::<Chunk>(16);

    // Channel for TTS job results
    let (tts_tx, mut tts_rx) = mpsc::channel::<TtsJobResult>(16);

    // Clone state for async tasks
    let inflight = state.tts_inflight.clone();
    let max_pending = state.tts_inflight_limit;
    let include_text = state.include_text;

    // Spawn the chunker task
    tokio::spawn(crate::chunker::run_chunker(
        chunk_in_rx,
        chunk_out_tx,
        config.clone(),
    ));

    // Spawn TTS processing task
    let tts = state.tts.clone();
    let server_engine = state.engine;
    let config_for_tts = config.clone();
    let selected_engine_for_tts = selected_engine.clone();
    let inflight_for_tts = inflight.clone();
    let tts_sender = tts_tx.clone();
    let (tts_done_tx, tts_done_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let mut pending: FuturesUnordered<BoxFuture<'static, TtsJobResult>> =
            FuturesUnordered::new();
        let mut chunk_rx_closed = false;

        loop {
            if chunk_rx_closed && pending.is_empty() {
                break;
            }

            tokio::select! {
                maybe_chunk = chunk_out_rx.recv(), if !chunk_rx_closed && pending.len() < max_pending => {
                    match maybe_chunk {
                        Some(chunk) => {
                            debug!(chunk_id = chunk.id, "processing chunk");
                            let tts = tts.clone();
                            let config_for_tts = config_for_tts.clone();
                            let selected_engine_for_tts = selected_engine_for_tts.clone();
                            let inflight_for_tts = inflight_for_tts.clone();

                            let fut = async move {
                                // Acquire permit for concurrent request limiting
                                let permit = inflight_for_tts.acquire_owned().await;
                                let permit = match permit {
                                    Ok(permit) => permit,
                                    Err(_) => {
                                        return TtsJobResult {
                                            id: chunk.id,
                                            result: Err("tts inflight limiter closed".to_string()),
                                        };
                                    }
                                };

                                // Build TTS request from config
                                let req = {
                                    let engine = *selected_engine_for_tts.read().await;
                                    if engine != server_engine {
                                        return TtsJobResult {
                                            id: chunk.id,
                                            result: Err(format!(
                                                "requested engine '{}' is not available on this server (serving '{}')",
                                                engine.as_str(),
                                                server_engine.as_str()
                                            )),
                                        };
                                    }

                                    let cfg = config_for_tts.read().await;
                                    TtsRequest {
                                        id: chunk.id,
                                        text: chunk.text,
                                        voice_path: cfg.voice_path.clone(),
                                        speed: cfg.speed,
                                        language_id: cfg.language_id.clone(),
                                        generation_config: Some(cfg.generation.clone()),
                                        streaming: false,
                                    }
                                };

                                // Perform TTS synthesis
                                let result = tts
                                    .synthesize(req)
                                    .await
                                    .map_err(|err| err.to_string());

                                drop(permit);
                                TtsJobResult { id: chunk.id, result }
                            };
                            pending.push(fut.boxed());
                        }
                        None => {
                            debug!("chunker channel closed");
                            chunk_rx_closed = true;
                        }
                    }
                }
                job = pending.next(), if !pending.is_empty() => {
                    if let Some(job) = job {
                        if let Err(e) = tts_sender.send(job).await {
                            warn!(error = %e, "failed to send tts job result");
                        }
                    }
                }
            }
        }
        drop(tts_sender);
        let _ = tts_done_tx.send(());
        debug!("tts processing task ended");
    });
    drop(tts_tx);

    // Spawn TTS output ordering task
    let out_tx_clone = out_tx.clone();
    let (tts_output_done_tx, tts_output_done_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let mut pending: HashMap<u64, Result<TtsResponse, String>> = HashMap::new();
        let mut next_id = 1u64;

        while let Some(job) = tts_rx.recv().await {
            pending.insert(job.id, job.result);

            // Emit in-order
            while let Some(result) = pending.remove(&next_id) {
                match result {
                    Ok(response) => {
                        debug!(chunk_id = %response.id, "sending audio frame");
                        let frame = build_audio_frame(&response, next_id, include_text);
                        if let Err(e) = out_tx_clone.send(Message::Binary(frame)).await {
                            warn!(error = %e, "failed to send audio frame");
                            break;
                        }
                    }
                    Err(message) => {
                        warn!(error = %message, "tts error");
                        let error_msg = serde_json::to_string(&ServerMessage::Error { message })
                            .unwrap_or_else(|_| {
                                "{\"type\":\"error\",\"message\":\"tts failure\"}".to_string()
                            });
                        if let Err(e) = out_tx_clone.send(Message::Text(error_msg)).await {
                            warn!(error = %e, "failed to send error message");
                            break;
                        }
                    }
                }
                next_id += 1;
            }
        }
        let _ = tts_output_done_tx.send(());
        debug!("tts output ordering task ended");
    });

    // Split WebSocket into sender and receiver
    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Spawn WebSocket writer task
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            if ws_sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Send ready message to client
    let ready_msg = serde_json::to_string(&ServerMessage::Ready {
        session_id: session_id.clone(),
    })
    .unwrap_or_else(|_| "{\"type\":\"ready\"}".to_string());

    if let Err(e) = out_tx.send(Message::Text(ready_msg)).await {
        error!(error = %e, "failed to send ready message");
        return;
    }

    // Main message processing loop
    let mut stop_requested = false;
    let mut tts_done_rx = tts_done_rx.fuse();
    let mut tts_output_done_rx = tts_output_done_rx.fuse();

    'recv: loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                let msg = match msg {
                    Some(Ok(msg)) => msg,
                    Some(Err(e)) => {
                        warn!(error = %e, "websocket error");
                        break 'recv;
                    }
                    None => {
                        debug!("websocket closed by client");
                        break 'recv;
                    }
                };

                match msg {
                    Message::Text(text) => {
                        match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(ClientMessage::Text { text }) => {
                                if !stop_requested {
                                    debug!(text_len = text.len(), "received text");
                                    if let Err(e) = chunk_in_tx.send(ChunkerInput::Text(text)).await {
                                        warn!(error = %e, "failed to send text to chunker");
                                    }
                                }
                            }
                            Ok(ClientMessage::Flush) => {
                                if !stop_requested {
                                    debug!("received flush");
                                    if let Err(e) = chunk_in_tx.send(ChunkerInput::Flush).await {
                                        warn!(error = %e, "failed to send flush to chunker");
                                    }
                                }
                            }
                            Ok(ClientMessage::Stop) => {
                                if !stop_requested {
                                    info!("received stop request");
                                    if let Err(e) = chunk_in_tx.send(ChunkerInput::Stop).await {
                                        warn!(error = %e, "failed to send stop to chunker");
                                    }
                                    stop_requested = true;
                                }
                            }
                            Ok(ClientMessage::Config {
                                engine,
                                voice_path,
                                speed,
                                language_id,
                                min_chars,
                                min_words,
                                max_chars,
                                max_delay_ms,
                            }) => {
                                if !stop_requested {
                                    info!("received config update");
                                    if let Some(engine) = engine {
                                        *selected_engine.write().await = engine;
                                        if engine != state.engine {
                                            warn!(
                                                requested_engine = engine.as_str(),
                                                active_engine = state.engine.as_str(),
                                                "client requested engine not active on this server"
                                            );
                                        }
                                    }
                                    let mut cfg = config.write().await;
                                    if let Some(voice_path) = voice_path {
                                        cfg.voice_path = Some(voice_path);
                                    }
                                    if let Some(speed) = speed {
                                        cfg.speed = speed.clamp(0.5, 2.0);
                                    }
                                    if let Some(language_id) = language_id {
                                        cfg.language_id = Some(language_id);
                                    }
                                    if let Some(min_chars) = min_chars {
                                        cfg.chunker.min_chars = min_chars;
                                    }
                                    if let Some(min_words) = min_words {
                                        cfg.chunker.min_words = min_words.max(1);
                                    }
                                    if let Some(max_chars) = max_chars {
                                        cfg.chunker.max_chars = max_chars;
                                    }
                                    if let Some(max_delay_ms) = max_delay_ms {
                                        cfg.chunker.max_delay_ms = max_delay_ms;
                                    }
                                    debug!(config = ?cfg, "updated config");
                                }
                            }
                            Err(err) => {
                                warn!(error = %err, "invalid message received");
                                let error_msg = serde_json::to_string(&ServerMessage::Error {
                                    message: format!("invalid message: {err}"),
                                })
                                .unwrap_or_else(|_| {
                                    "{\"type\":\"error\",\"message\":\"invalid message\"}".to_string()
                                });
                                if let Err(e) = out_tx.send(Message::Text(error_msg)).await {
                                    warn!(error = %e, "failed to send error message");
                                }
                            }
                        }
                    }
                    Message::Close(_) => {
                        debug!("received close frame");
                        break 'recv;
                    }
                    Message::Binary(_) => {
                        // Client shouldn't send binary, ignore
                        debug!("received unexpected binary message");
                    }
                    Message::Ping(data) => {
                        // Respond with pong
                        if let Err(e) = out_tx.send(Message::Pong(data)).await {
                            warn!(error = %e, "failed to send pong");
                        }
                    }
                    Message::Pong(_) => {
                        // Received pong, connection is alive
                    }
                }
            }
            _ = &mut tts_done_rx, if stop_requested && !tts_done_rx.is_terminated() => {}
            _ = &mut tts_output_done_rx, if stop_requested && !tts_output_done_rx.is_terminated() => {}
        }

        if stop_requested && tts_done_rx.is_terminated() && tts_output_done_rx.is_terminated() {
            break 'recv;
        }
    }

    // Send done message
    let done_msg = serde_json::to_string(&ServerMessage::Done)
        .unwrap_or_else(|_| "{\"type\":\"done\"}".to_string());

    if let Err(e) = out_tx.send(Message::Text(done_msg)).await {
        warn!(error = %e, "failed to send done message");
    }

    drop(out_tx);
    drop(chunk_in_tx);

    // Wait for writer to complete
    let _ = writer.await;
    info!(session_id = %session_id, "websocket connection closed");
}

/// Audio frame header for binary transmission
#[derive(Serialize)]
struct AudioHeaderRef<'a> {
    chunk_id: u64,
    sample_rate: u32,
    channels: u16,
    format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
}

/// Build a binary audio frame with header and PCM data
///
/// Frame format:
/// - 4 bytes: header length (big-endian u32)
/// - N bytes: JSON header
/// - M bytes: PCM audio data (i16 little-endian)
fn build_audio_frame(response: &TtsResponse, _chunk_id: u64, include_text: bool) -> Vec<u8> {
    // Convert f32 samples to i16 bytes
    let pcm_bytes = pcm_f32_to_bytes(&response.pcm);

    let header = AudioHeaderRef {
        chunk_id: response.id,
        sample_rate: response.sample_rate,
        channels: response.channels,
        format: &response.format.to_string(),
        text: if include_text && !response.text.is_empty() {
            Some(&response.text)
        } else {
            None
        },
    };

    let header_json = serde_json::to_vec(&header).unwrap_or_default();
    let mut frame = Vec::with_capacity(4 + header_json.len() + pcm_bytes.len());
    frame.extend_from_slice(&(header_json.len() as u32).to_be_bytes());
    frame.extend_from_slice(&header_json);
    frame.extend_from_slice(&pcm_bytes);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::{AudioFormat, GenerationMetadata, TtsError};
    use async_trait::async_trait;

    struct MockTtsEngine;

    #[async_trait]
    impl TtsEngine for MockTtsEngine {
        async fn synthesize(&self, _req: TtsRequest) -> Result<TtsResponse, TtsError> {
            Ok(TtsResponse {
                id: 1,
                sample_rate: 32000,
                channels: 1,
                format: AudioFormat::PcmI16,
                pcm: vec![0.0f32; 512],
                text: "test".to_string(),
                duration_secs: 0.016,
                num_samples: 512,
                metadata: GenerationMetadata::default(),
            })
        }

        async fn synthesize_streaming(
            &self,
            _req: TtsRequest,
        ) -> Result<tokio::sync::mpsc::Receiver<Result<crate::tts::TtsChunk, TtsError>>, TtsError>
        {
            let (tx, rx) = tokio::sync::mpsc::channel(4);
            tokio::spawn(async move {
                let _ = tx
                    .send(Ok(crate::tts::TtsChunk {
                        sequence: 0,
                        pcm: vec![0.0f32; 256],
                        is_final: true,
                        text: Some("test".to_string()),
                    }))
                    .await;
            });
            Ok(rx)
        }

        async fn synthesize_batch(
            &self,
            requests: Vec<TtsRequest>,
        ) -> Vec<Result<TtsResponse, TtsError>> {
            let mut results = Vec::with_capacity(requests.len());
            for req in requests {
                results.push(self.synthesize(req).await);
            }
            results
        }

        fn info(&self) -> crate::tts::EngineInfo {
            crate::tts::EngineInfo::default()
        }

        async fn health_check(&self) -> Result<(), TtsError> {
            Ok(())
        }
    }

    #[test]
    fn test_app_state_creation() {
        let tts: Arc<dyn TtsEngine> = Arc::new(MockTtsEngine);
        let config = StreamConfig::default();
        let state = AppState::new_with_engine(tts, EngineId::Soprano, config, 4, false);

        assert_eq!(state.engine, EngineId::Soprano);
        assert_eq!(state.tts_inflight_limit, 4);
        assert!(!state.include_text);
    }

    #[test]
    fn test_build_audio_frame() {
        let response = TtsResponse {
            id: 42,
            sample_rate: 32000,
            channels: 1,
            format: AudioFormat::PcmI16,
            pcm: vec![0.5f32, -0.5f32, 0.25f32, -0.25f32],
            text: "hello".to_string(),
            duration_secs: 0.001,
            num_samples: 4,
            metadata: GenerationMetadata::default(),
        };

        let frame = build_audio_frame(&response, 42, true);

        // Check header length prefix
        let header_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        assert!(header_len > 0);
        assert!(frame.len() > 4 + header_len);

        // Check PCM data is included (4 f32 samples = 8 bytes as i16)
        assert!(frame.len() > 4 + header_len);
    }

    #[test]
    fn test_build_audio_frame_without_text() {
        let response = TtsResponse {
            id: 42,
            sample_rate: 32000,
            channels: 1,
            format: AudioFormat::PcmI16,
            pcm: vec![0.5f32, -0.5f32],
            text: "hello".to_string(),
            duration_secs: 0.001,
            num_samples: 2,
            metadata: GenerationMetadata::default(),
        };

        let frame = build_audio_frame(&response, 42, false);

        // Header should not contain text field
        let header_len = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let header_json = String::from_utf8_lossy(&frame[4..4 + header_len]);
        assert!(!header_json.contains("text"));
    }

    #[test]
    fn test_pcm_f32_to_bytes() {
        let samples = vec![1.0f32, -1.0f32, 0.0f32, 0.5f32, -0.5f32];
        let bytes = pcm_f32_to_bytes(&samples);

        // Each f32 becomes 2 bytes as i16
        assert_eq!(bytes.len(), samples.len() * 2);

        // Check that 1.0 maps to 32767 (max i16)
        let max_i16 = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(max_i16, 32767);

        // Check that -1.0 maps to -32767 (min i16)
        let min_i16 = i16::from_le_bytes([bytes[2], bytes[3]]);
        assert_eq!(min_i16, -32767);
    }
}

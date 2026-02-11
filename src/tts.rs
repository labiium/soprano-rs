//! Text-to-Speech engine for Soprano TTS
//!
//! Provides a complete TTS pipeline with batch and streaming inference support.
//! Uses Candle ML framework for efficient model inference.

use crate::config::{DecoderConfig, GenerationConfig};
use crate::decoder::SopranoDecoder;
use crate::model::SopranoModel;
use crate::normalization::clean_text;
use crate::splitter::split_and_recombine_text;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
// Note: we use tokio::sync::Mutex for async compatibility
use tokio::sync::Mutex as AsyncMutex;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{debug, info, instrument};

/// Save tensor as .npy file for debugging
fn save_tensor_npy(tensor: &Tensor, filename: &str) -> Result<(), TtsError> {
    let path = Path::new(filename);
    let data: Vec<f32> = tensor.flatten_all()
        .map_err(|e| TtsError::DecoderError(e.to_string()))?
        .to_vec1()
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    let shape = tensor.dims();

    // Create numpy .npy format header
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {:?}, }}",
        shape
    );
    let header_len = header.len();
    let padding = (16 - ((10 + header_len) % 16)) % 16;
    let total_header_len = 10 + header_len + padding;

    let mut file = File::create(path)
        .map_err(|e| TtsError::DecoderError(format!("Failed to create {}: {}", filename, e)))?;

    // Write magic number and version
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00])
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    
    // Write header length (little endian u16)
    file.write_all(&(total_header_len as u16).to_le_bytes())
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    
    // Write header
    file.write_all(header.as_bytes())
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    
    // Write padding
    file.write_all(&vec![0x20; padding])
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    
    // Write data
    let data_bytes: Vec<u8> = data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    file.write_all(&data_bytes)
        .map_err(|e| TtsError::DecoderError(e.to_string()))?;
    
    Ok(())
}

fn save_hidden_states_debug(hidden_states: &Tensor) -> Result<(), TtsError> {
    save_tensor_npy(hidden_states, "rust_hidden_states.npy")
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during TTS synthesis
#[derive(Error, Debug, Clone)]
pub enum TtsError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Decoder error: {0}")]
    DecoderError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Text normalization failed: {0}")]
    NormalizationError(String),

    #[error("Inference timeout after {0}ms")]
    Timeout(u64),

    #[error("Worker pool exhausted")]
    WorkerPoolExhausted,

    #[error("Channel closed")]
    ChannelClosed,

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Batch processing error: {0}")]
    BatchError(String),
}

impl From<candle_core::Error> for TtsError {
    fn from(err: candle_core::Error) -> Self {
        TtsError::ModelError(err.to_string())
    }
}

impl From<std::io::Error> for TtsError {
    fn from(err: std::io::Error) -> Self {
        TtsError::IoError(err.to_string())
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// TTS synthesis request
#[derive(Clone, Debug)]
pub struct TtsRequest {
    /// Unique request identifier (monotonically increasing for streaming chunks)
    pub id: u64,
    /// Text to synthesize
    pub text: String,
    /// Path to voice model (optional, for multi-voice support)
    pub voice_path: Option<String>,
    /// Speech speed multiplier (0.5 - 2.0)
    pub speed: f32,
    /// Language identifier (e.g., "en", "fr")
    pub language_id: Option<String>,
    /// Generation configuration
    pub generation_config: Option<GenerationConfig>,
    /// Whether to enable streaming
    pub streaming: bool,
}

impl TtsRequest {
    /// Create a new TTS request with defaults
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: 0,
            text: text.into(),
            voice_path: None,
            speed: 1.0,
            language_id: Some("en".to_string()),
            generation_config: None,
            streaming: false,
        }
    }

    /// Set the voice path
    pub fn with_voice(mut self, path: impl Into<String>) -> Self {
        self.voice_path = Some(path.into());
        self
    }

    /// Set the speech speed
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Set the language
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language_id = Some(lang.into());
        self
    }

    /// Set generation configuration
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = Some(config);
        self
    }

    /// Enable streaming mode
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Set custom request ID
    pub fn with_id(mut self, id: u64) -> Self {
        self.id = id;
        self
    }
}

impl Default for TtsRequest {
    fn default() -> Self {
        Self::new("")
    }
}

/// Convert config::GenerationConfig to model::GenerationConfig
fn to_model_generation_config(cfg: &GenerationConfig) -> crate::model::GenerationConfig {
    crate::model::GenerationConfig {
        max_new_tokens: cfg.max_new_tokens,
        temperature: cfg.temperature,
        top_p: cfg.top_p,
        repetition_penalty: cfg.repetition_penalty,
        min_new_tokens: cfg.min_new_tokens,
    }
}

/// Audio format for TTS output
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AudioFormat {
    PcmI16,
    #[default]
    PcmF32,
    Wav,
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioFormat::PcmI16 => write!(f, "pcm_s16le"),
            AudioFormat::PcmF32 => write!(f, "pcm_f32le"),
            AudioFormat::Wav => write!(f, "wav"),
        }
    }
}

/// TTS synthesis response
#[derive(Clone, Debug)]
pub struct TtsResponse {
    /// Request identifier (chunk ID for streaming)
    pub id: u64,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u16,
    /// Audio format
    pub format: AudioFormat,
    /// PCM audio data (interleaved for multi-channel)
    pub pcm: Vec<f32>,
    /// Original text (for alignment)
    pub text: String,
    /// Duration of audio in seconds
    pub duration_secs: f32,
    /// Number of audio samples
    pub num_samples: usize,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Metadata about the generation process
#[derive(Clone, Debug, Default)]
pub struct GenerationMetadata {
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Number of sentences processed
    pub sentences_processed: usize,
    /// Total processing time
    pub processing_time_ms: u64,
    /// Time spent in LLM inference
    pub llm_time_ms: u64,
    /// Time spent in decoder
    pub decoder_time_ms: u64,
    /// Finish reason
    pub finish_reason: String,
}

impl TtsResponse {
    /// Create a new TTS response
    pub fn new(id: u64, sample_rate: u32, pcm: Vec<f32>) -> Self {
        let num_samples = pcm.len();
        let duration_secs = num_samples as f32 / sample_rate as f32;

        Self {
            id,
            sample_rate,
            channels: 1,
            format: AudioFormat::default(),
            pcm,
            text: String::new(),
            duration_secs,
            num_samples,
            metadata: GenerationMetadata::default(),
        }
    }

    /// Get audio as i16 samples
    pub fn pcm_i16(&self) -> Vec<i16> {
        self.pcm
            .iter()
            .map(|&s| {
                // Map [-1.0, 1.0] to [-32768, 32767] with saturation.
                let v = (s.clamp(-1.0, 1.0) * 32768.0).round() as i32;
                v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect()
    }

    /// Get audio as bytes (f32 little-endian)
    pub fn pcm_as_bytes(&self) -> Vec<u8> {
        self.pcm
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect()
    }

    /// Get audio as bytes (i16 little-endian)
    pub fn pcm_i16_as_bytes(&self) -> Vec<u8> {
        self.pcm_i16()
            .iter()
            .flat_map(|&s| s.to_le_bytes())
            .collect()
    }
}

/// Streaming TTS chunk
#[derive(Clone, Debug)]
pub struct TtsChunk {
    /// Chunk sequence number
    pub sequence: usize,
    /// PCM audio data for this chunk
    pub pcm: Vec<f32>,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Text corresponding to this chunk (if alignment enabled)
    pub text: Option<String>,
}

// =============================================================================
// TTS Engine Trait
// =============================================================================

/// Trait for TTS engines
#[async_trait]
pub trait TtsEngine: Send + Sync {
    /// Synthesize speech from text
    async fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError>;

    /// Synthesize speech with streaming output
    async fn synthesize_streaming(
        &self,
        req: TtsRequest,
    ) -> Result<mpsc::Receiver<Result<TtsChunk, TtsError>>, TtsError>;

    /// Synthesize multiple texts in batch
    async fn synthesize_batch(
        &self,
        requests: Vec<TtsRequest>,
    ) -> Vec<Result<TtsResponse, TtsError>>;

    /// Get engine information
    fn info(&self) -> EngineInfo;

    /// Check if engine is healthy
    async fn health_check(&self) -> Result<(), TtsError>;
}

/// Engine information
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub name: String,
    pub version: String,
    pub sample_rate: u32,
    pub max_text_length: usize,
    pub supports_streaming: bool,
    pub supports_batching: bool,
}

impl Default for EngineInfo {
    fn default() -> Self {
        Self {
            name: "SopranoTTS".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            sample_rate: 32000,
            max_text_length: 10000,
            supports_streaming: true,
            supports_batching: true,
        }
    }
}

// =============================================================================
// Decoder Worker Pool
// =============================================================================

/// Worker task for the decoder pool
#[derive(Debug)]
struct DecoderTask {
    hidden_states: Tensor,
    trim_tokens: usize,
    response_tx: oneshot::Sender<Result<Tensor, TtsError>>,
}

/// Worker for parallel decoder processing
struct DecoderWorker {
    decoder: SopranoDecoder,
    #[allow(dead_code)]
    device: Device,
}

/// Pool of decoder workers for parallel processing
pub struct DecoderWorkerPool {
    task_tx: mpsc::Sender<DecoderTask>,
    semaphore: Arc<Semaphore>,
    decoder_device: Device,
}

impl DecoderWorkerPool {
    /// Create a new worker pool with the specified number of workers
    pub fn new(
        num_workers: usize,
        model_path: &Path,
        decoder_config: &DecoderConfig,
        _device: &Device,
    ) -> Result<Self, TtsError> {
        let (task_tx, task_rx) = mpsc::channel::<DecoderTask>(num_workers * 2);
        let task_rx = Arc::new(AsyncMutex::new(task_rx));

        // The ISTFT step is currently CPU-based. Running the full decoder on CPU
        // avoids expensive GPU->CPU transfers of large spectrogram tensors.
        let decoder_device = Device::Cpu;

        let decoder_pth = model_path.join("decoder.pth");
        if !decoder_pth.exists() {
            return Err(TtsError::DecoderError(format!(
                "missing decoder.pth at {}",
                decoder_pth.display()
            )));
        }

        // Create workers
        for i in 0..num_workers {
            let vb = candle_nn::VarBuilder::from_pth(&decoder_pth, DType::F32, &decoder_device)
                .map_err(|e| TtsError::DecoderError(format!("failed to load decoder.pth: {e}")))?;

            let decoder = SopranoDecoder::new(
                decoder_config.num_input_channels,
                decoder_config.decoder_num_layers,
                decoder_config.decoder_dim,
                decoder_config.decoder_intermediate_dim,
                decoder_config.hop_length,
                decoder_config.n_fft,
                decoder_config.upscale,
                decoder_config.dw_kernel,
                vb,
            ).map_err(|e| TtsError::DecoderError(format!("Failed to create decoder {}: {}", i, e)))?;

            let worker = DecoderWorker {
                decoder,
                device: decoder_device.clone(),
            };

            let worker_rx = Arc::clone(&task_rx);
            tokio::spawn(async move {
                loop {
                    let task = {
                        let mut rx = worker_rx.lock().await;
                        rx.recv().await
                    };
                    
                    match task {
                        Some(task) => {
                            let result = worker
                                .decoder
                                .decode(&task.hidden_states, task.trim_tokens)
                                .map_err(|e| TtsError::DecoderError(e.to_string()));
                            let _ = task.response_tx.send(result);
                        }
                        None => break, // Channel closed
                    }
                }
            });
        }

        Ok(Self {
            task_tx,
            semaphore: Arc::new(Semaphore::new(num_workers)),
            decoder_device,
        })
    }

    /// Submit a decode task to the pool
    pub async fn decode(
        &self,
        hidden_states: Tensor,
        trim_tokens: usize,
    ) -> Result<Tensor, TtsError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| TtsError::WorkerPoolExhausted)?;

        let (tx, rx) = oneshot::channel();

        let hidden_states = hidden_states
            .to_device(&self.decoder_device)
            .map_err(|e| TtsError::DecoderError(e.to_string()))?;

        let task = DecoderTask {
            hidden_states,
            trim_tokens,
            response_tx: tx,
        };

        self.task_tx
            .send(task)
            .await
            .map_err(|_| TtsError::ChannelClosed)?;

        rx.await
            .map_err(|_| TtsError::ChannelClosed)?
    }
}

// =============================================================================
// Soprano TTS Engine
// =============================================================================

/// Configuration for the Soprano TTS Engine
#[derive(Clone, Debug)]
pub struct SopranoEngineConfig {
    /// Path to model directory
    pub model_path: std::path::PathBuf,
    /// Device to use for inference
    pub device: Device,
    /// Number of decoder workers
    pub num_workers: usize,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Default generation config
    pub generation_config: GenerationConfig,
    /// Decoder configuration
    pub decoder_config: DecoderConfig,
    /// Maximum text length
    pub max_text_length: usize,
    /// Inference timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for SopranoEngineConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models"),
            device: Device::Cpu,
            num_workers: 2,
            sample_rate: 32000,
            generation_config: GenerationConfig::default(),
            decoder_config: DecoderConfig::default(),
            max_text_length: 10000,
            timeout_ms: 60000,
        }
    }
}

/// Soprano TTS Engine implementation
pub struct SopranoTtsEngine {
    /// The LLM model for generating hidden states
    model: Arc<AsyncMutex<SopranoModel>>,
    /// Decoder worker pool
    decoder_pool: Arc<DecoderWorkerPool>,
    /// Device for inference
    device: Device,
    /// Engine configuration
    config: SopranoEngineConfig,
    /// Sample rate for output
    sample_rate: u32,
}

impl SopranoTtsEngine {
    /// Create a new Soprano TTS Engine
    pub async fn new(config: SopranoEngineConfig) -> Result<Self, TtsError> {
        info!("Initializing Soprano TTS Engine");
        info!("Model path: {:?}", config.model_path);
        info!("Device: {:?}", config.device);
        info!("Workers: {}", config.num_workers);

        // Load the model
        let model = SopranoModel::from_path(&config.model_path, config.device.clone())
            .map_err(|e| TtsError::ModelError(format!("Failed to load model: {}", e)))?;

        info!("Model loaded successfully");

        // Create decoder worker pool
        let decoder_pool = DecoderWorkerPool::new(
            config.num_workers,
            &config.model_path,
            &config.decoder_config,
            &config.device,
        )?;

        info!("Decoder worker pool created with {} workers", config.num_workers);

        Ok(Self {
            model: Arc::new(AsyncMutex::new(model)),
            decoder_pool: Arc::new(decoder_pool),
            device: config.device.clone(),
            config: config.clone(),
            sample_rate: config.sample_rate,
        })
    }

    /// Create from a model path string
    pub async fn from_path(model_path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let config = SopranoEngineConfig {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        };
        Self::new(config).await
    }

    /// Preprocess text for the model
    fn preprocess_text(&self, text: &str, _sentence_idx: usize) -> String {
        format!("[STOP][TEXT]{}[START]", text.trim())
    }

    /// Process a single sentence through the TTS pipeline
    #[instrument(skip(self, text), fields(text_len = text.len()))]
    async fn process_sentence(
        &self,
        text: &str,
        gen_config: &GenerationConfig,
    ) -> Result<(Vec<f32>, GenerationMetadata), TtsError> {
        let start_time = Instant::now();
        let mut metadata = GenerationMetadata::default();

        // Preprocess text
        let prompt = self.preprocess_text(text, 0);
        debug!("Preprocessed prompt: {}", prompt);

        // Convert to model's GenerationConfig
        let model_config = to_model_generation_config(gen_config);

        // Generate hidden states using LLM
        let llm_start = Instant::now();
        let generation_result = {
            let mut model = self.model.lock().await;
            model.generate(&prompt, &model_config)
                .map_err(|e| TtsError::ModelError(e.to_string()))?
        };
        metadata.llm_time_ms = llm_start.elapsed().as_millis() as u64;
        metadata.tokens_generated = generation_result.token_count;
        metadata.finish_reason = format!("{:?}", generation_result.finish_reason);

        debug!("Generated {} tokens in {}ms", 
            generation_result.token_count, 
            metadata.llm_time_ms
        );

        // Decode hidden states to audio
        let decoder_start = Instant::now();

        // Hidden states come as (T, H). Decoder expects (B, H, T).
        let token_len = generation_result
            .hidden_states
            .dim(0)
            .map_err(|e| TtsError::DecoderError(e.to_string()))?;

        // Debug: save hidden states from LLM
        let debug = std::env::var("SOPRANO_DEBUG_TENSORS").is_ok();
        if debug {
            let _ = save_hidden_states_debug(&generation_result.hidden_states);
        }

        let hs = generation_result
            .hidden_states
            .unsqueeze(0)
            .and_then(|t| t.transpose(1, 2))
            .map_err(|e| TtsError::DecoderError(e.to_string()))?;

        let audio_tensor = self
            .decoder_pool
            .decode(hs, 0)
            .await?;
        metadata.decoder_time_ms = decoder_start.elapsed().as_millis() as u64;

        // Decoder returns (B, L). We use batch size 1 here.
        let audio_1d = audio_tensor
            .squeeze(0)
            .map_err(|e| TtsError::DecoderError(e.to_string()))?;

        // Match Python reference: keep last (T * TOKEN_SIZE - TOKEN_SIZE) samples.
        // This drops the first token-sized chunk.
        const TOKEN_SIZE: usize = 2048;
        let desired = token_len.saturating_mul(TOKEN_SIZE).saturating_sub(TOKEN_SIZE);
        let audio_1d = if desired > 0 {
            let len = audio_1d
                .dim(0)
                .map_err(|e| TtsError::DecoderError(e.to_string()))?;
            if len > desired {
                audio_1d
                    .narrow(0, len - desired, desired)
                    .map_err(|e| TtsError::DecoderError(e.to_string()))?
            } else {
                audio_1d
            }
        } else {
            audio_1d
        };

        let audio: Vec<f32> = audio_1d
            .to_vec1()
            .map_err(|e| TtsError::DecoderError(e.to_string()))?;

        metadata.processing_time_ms = start_time.elapsed().as_millis() as u64;
        metadata.sentences_processed = 1;

        debug!("Sentence processed in {}ms", metadata.processing_time_ms);

        Ok((audio, metadata))
    }

    /// Apply speed adjustment to audio
    fn apply_speed(&self, audio: Vec<f32>, speed: f32) -> Vec<f32> {
        if (speed - 1.0).abs() < 0.01 {
            return audio;
        }

        // Simple linear interpolation for speed adjustment
        let new_len = (audio.len() as f32 / speed) as usize;
        let mut result = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = (i as f32 * speed) as usize;
            if src_idx < audio.len() {
                result.push(audio[src_idx]);
            }
        }

        result
    }

    /// Concatenate multiple audio chunks
    fn concatenate_audio(chunks: Vec<Vec<f32>>) -> Vec<f32> {
        let total_len = chunks.iter().map(|c| c.len()).sum();
        let mut result = Vec::with_capacity(total_len);
        for chunk in chunks {
            result.extend(chunk);
        }
        result
    }

    /// Normalize audio to [-1, 1] range with headroom and noise reduction
    fn normalize_audio(audio: Vec<f32>) -> Vec<f32> {
        if audio.is_empty() {
            return audio;
        }
        
        // Find peak amplitude
        let max_abs = audio
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));
        
        if max_abs > 0.0 {
            // Normalize with 6dB headroom (multiply by 0.5)
            // This provides more headroom to prevent clipping
            let target_peak = 0.5; // -6dB
            let scale = target_peak / max_abs;
            
            // Apply scaling with soft clipping
            audio.iter().map(|&s| {
                let scaled = s * scale;
                // Soft clipping using tanh for smoother limiting
                // Only apply to values near the limits
                if scaled.abs() > 0.8 {
                    scaled.tanh()
                } else {
                    scaled
                }
            }).collect()
        } else {
            audio
        }
    }

    /// Merge metadata from multiple chunks
    fn merge_metadata(metadatas: Vec<GenerationMetadata>) -> GenerationMetadata {
        let mut result = GenerationMetadata::default();
        for m in metadatas {
            result.tokens_generated += m.tokens_generated;
            result.sentences_processed += m.sentences_processed;
            result.llm_time_ms += m.llm_time_ms;
            result.decoder_time_ms += m.decoder_time_ms;
            result.processing_time_ms = result.processing_time_ms.max(m.processing_time_ms);
        }
        result.finish_reason = "Complete".to_string();
        result
    }
}

#[async_trait]
impl TtsEngine for SopranoTtsEngine {
    #[instrument(skip(self, req))]
    async fn synthesize(&self, req: TtsRequest) -> Result<TtsResponse, TtsError> {
        let start_time = Instant::now();
        let request_id = req.id;

        info!("Processing TTS request: {}", request_id);

        // Validate input
        if req.text.is_empty() {
            return Err(TtsError::InvalidInput("Empty text provided".to_string()));
        }

        if req.text.len() > self.config.max_text_length {
            return Err(TtsError::InvalidInput(format!(
                "Text too long: {} > {}",
                req.text.len(),
                self.config.max_text_length
            )));
        }

        // Get generation config
        let gen_config = req.generation_config.unwrap_or_else(|| self.config.generation_config.clone());

        // Clean and normalize text
        let clean_start = Instant::now();
        let cleaned_text = clean_text(&req.text);
        debug!("Text cleaned in {:?}", clean_start.elapsed());

        // Split into sentences
        let sentences = split_and_recombine_text(&cleaned_text, 200, 400);
        debug!("Split into {} sentences/chunks", sentences.len());

        if sentences.is_empty() {
            return Err(TtsError::InvalidInput("No valid text to synthesize".to_string()));
        }

        // Process each sentence
        let mut audio_chunks = Vec::new();
        let mut all_metadata = Vec::new();

        for (idx, sentence) in sentences.iter().enumerate() {
            debug!("Processing sentence {}/{}: '{}'", idx + 1, sentences.len(), sentence);

            let (audio, metadata) = self.process_sentence(sentence, &gen_config).await?;
            audio_chunks.push(audio);
            all_metadata.push(metadata);
        }

        // Concatenate audio
        let mut final_audio = Self::concatenate_audio(audio_chunks);
        
        // Normalize audio to prevent clipping and set proper volume
        final_audio = Self::normalize_audio(final_audio);

        // Apply speed adjustment
        if req.speed != 1.0 {
            final_audio = self.apply_speed(final_audio, req.speed);
        }

        // Create response
        let mut response = TtsResponse::new(request_id, self.sample_rate, final_audio);
        response.text = req.text;
        response.metadata = Self::merge_metadata(all_metadata);

        info!(
            "TTS request {} completed: {} samples, {:.2}s audio in {:?}",
            request_id,
            response.num_samples,
            response.duration_secs,
            start_time.elapsed()
        );

        Ok(response)
    }

    #[instrument(skip(self, req))]
    async fn synthesize_streaming(
        &self,
        req: TtsRequest,
    ) -> Result<mpsc::Receiver<Result<TtsChunk, TtsError>>, TtsError> {
        let (tx, rx) = mpsc::channel(4);
        let engine = Arc::new(self.clone_for_streaming()?);

        tokio::spawn(async move {
            if let Err(e) = engine.streaming_worker(req, tx.clone()).await {
                let _ = tx.send(Err(e)).await;
            }
        });

        Ok(rx)
    }

    #[instrument(skip(self, requests))]
    async fn synthesize_batch(
        &self,
        requests: Vec<TtsRequest>,
    ) -> Vec<Result<TtsResponse, TtsError>> {
        let mut results = Vec::with_capacity(requests.len());

        for req in requests {
            let result = self.synthesize(req).await;
            results.push(result);
        }

        results
    }

    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "SopranoTtsEngine".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            sample_rate: self.sample_rate,
            max_text_length: self.config.max_text_length,
            supports_streaming: true,
            supports_batching: true,
        }
    }

    async fn health_check(&self) -> Result<(), TtsError> {
        // Try to acquire a permit from the decoder pool
        let _permit = self
            .decoder_pool
            .semaphore
            .acquire()
            .await
            .map_err(|_| TtsError::WorkerPoolExhausted)?;

        // Check if model is accessible
        let _model = self.model.lock().await;

        Ok(())
    }
}

// Clone implementation for streaming support
impl SopranoTtsEngine {
    fn clone_for_streaming(&self) -> Result<Self, TtsError> {
        // Create a new model instance for streaming
        let model = SopranoModel::from_path(&self.config.model_path, self.device.clone())
            .map_err(|e| TtsError::ModelError(format!("Failed to clone model for streaming: {}", e)))?;

        Ok(Self {
            model: Arc::new(AsyncMutex::new(model)),
            decoder_pool: Arc::clone(&self.decoder_pool),
            device: self.device.clone(),
            config: self.config.clone(),
            sample_rate: self.sample_rate,
        })
    }

    async fn streaming_worker(
        &self,
        req: TtsRequest,
        tx: mpsc::Sender<Result<TtsChunk, TtsError>>,
    ) -> Result<(), TtsError> {
        let gen_config = req.generation_config.unwrap_or_else(|| self.config.generation_config.clone());

        // Clean and split text
        let cleaned_text = clean_text(&req.text);
        let sentences = split_and_recombine_text(&cleaned_text, 200, 400);

        for (idx, sentence) in sentences.iter().enumerate() {
            let is_final = idx == sentences.len() - 1;

            match self.process_sentence(sentence, &gen_config).await {
                Ok((audio, _metadata)) => {
                    let chunk = TtsChunk {
                        sequence: idx,
                        pcm: audio,
                        is_final,
                        text: Some(sentence.clone()),
                    };

                    if tx.send(Ok(chunk)).await.is_err() {
                        return Err(TtsError::ChannelClosed);
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.clone())).await;
                    return Err(e);
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// Builder Pattern
// =============================================================================

/// Builder for SopranoTtsEngine
pub struct SopranoTtsEngineBuilder {
    config: SopranoEngineConfig,
}

impl SopranoTtsEngineBuilder {
    /// Create a new builder
    pub fn new(model_path: impl AsRef<Path>) -> Self {
        Self {
            config: SopranoEngineConfig {
                model_path: model_path.as_ref().to_path_buf(),
                ..Default::default()
            },
        }
    }

    /// Set the device
    pub fn with_device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    /// Set the number of workers
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.config.num_workers = workers.max(1);
        self
    }

    /// Set the sample rate
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.config.sample_rate = sample_rate;
        self
    }

    /// Set the generation config
    pub fn with_generation_config(mut self, config: GenerationConfig) -> Self {
        self.config.generation_config = config;
        self
    }

    /// Set the decoder config
    pub fn with_decoder_config(mut self, config: DecoderConfig) -> Self {
        self.config.decoder_config = config;
        self
    }

    /// Set the max text length
    pub fn with_max_text_length(mut self, max_len: usize) -> Self {
        self.config.max_text_length = max_len;
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.timeout_ms = timeout_ms;
        self
    }

    /// Build the engine
    pub async fn build(self) -> Result<SopranoTtsEngine, TtsError> {
        SopranoTtsEngine::new(self.config).await
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_request_builder() {
        let req = TtsRequest::new("Hello world")
            .with_speed(1.5)
            .with_language("en")
            .with_streaming(true);

        assert_eq!(req.text, "Hello world");
        assert_eq!(req.speed, 1.5);
        assert_eq!(req.language_id, Some("en".to_string()));
        assert!(req.streaming);
    }

    #[test]
    fn test_tts_request_speed_clamping() {
        let req = TtsRequest::new("Test").with_speed(3.0);
        assert_eq!(req.speed, 2.0);

        let req = TtsRequest::new("Test").with_speed(0.1);
        assert_eq!(req.speed, 0.5);
    }

    #[test]
    fn test_audio_format_display() {
        assert_eq!(format!("{}", AudioFormat::PcmF32), "pcm_f32le");
        assert_eq!(format!("{}", AudioFormat::PcmI16), "pcm_s16le");
        assert_eq!(format!("{}", AudioFormat::Wav), "wav");
    }

    #[test]
    fn test_tts_response_pcm_i16() {
        let response = TtsResponse::new(1, 32000, vec![0.0, 0.5, -0.5, 1.0, -1.0]);
        let i16_samples = response.pcm_i16();

        assert_eq!(i16_samples[0], 0);
        assert!(i16_samples[1] > 0);
        assert!(i16_samples[2] < 0);
        assert_eq!(i16_samples[3], i16::MAX);
        assert_eq!(i16_samples[4], i16::MIN);
    }

    #[test]
    fn test_concatenate_audio() {
        let chunks = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0],
        ];

        let result = SopranoTtsEngine::concatenate_audio(chunks);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_merge_metadata() {
        let metadatas = vec![
            GenerationMetadata {
                tokens_generated: 10,
                sentences_processed: 1,
                llm_time_ms: 100,
                decoder_time_ms: 50,
                ..Default::default()
            },
            GenerationMetadata {
                tokens_generated: 20,
                sentences_processed: 1,
                llm_time_ms: 200,
                decoder_time_ms: 100,
                ..Default::default()
            },
        ];

        let merged = SopranoTtsEngine::merge_metadata(metadatas);
        assert_eq!(merged.tokens_generated, 30);
        assert_eq!(merged.sentences_processed, 2);
        assert_eq!(merged.llm_time_ms, 300);
        assert_eq!(merged.decoder_time_ms, 150);
    }

    #[test]
    fn test_tts_error_conversions() {
        let candle_err = candle_core::Error::Msg("test".to_string());
        let tts_err: TtsError = candle_err.into();
        assert!(matches!(tts_err, TtsError::ModelError(_)));

        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let tts_err: TtsError = io_err.into();
        assert!(matches!(tts_err, TtsError::IoError(_)));
    }

    #[test]
    fn test_engine_info_default() {
        let info = EngineInfo::default();
        assert_eq!(info.name, "SopranoTTS");
        assert_eq!(info.sample_rate, 32000);
        assert!(info.supports_streaming);
        assert!(info.supports_batching);
    }

    #[test]
    fn test_soprano_engine_config_default() {
        let config = SopranoEngineConfig::default();
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.sample_rate, 32000);
        assert_eq!(config.max_text_length, 10000);
        assert_eq!(config.timeout_ms, 60000);
    }
}

//! Text chunking module for streaming TTS
//!
//! This module provides efficient text chunking for streaming text-to-speech synthesis.
//! It buffers incoming text, detects sentence boundaries, and emits chunks of appropriate
//! size for the TTS engine while respecting timing and size constraints.
//!
//! # Key Features
//!
//! - **Sentence boundary detection**: Splits on punctuation marks for natural speech breaks
//! - **UTF-8 safe**: All operations respect Unicode character boundaries
//! - **Configurable**: Min/max characters, words, and timeout delays
//! - **Async/await**: Built on tokio channels for non-blocking operation
//! - **Timeout handling**: Emits partial chunks when delays are exceeded

use std::{sync::Arc, time::Duration};

use tokio::sync::{mpsc, RwLock};
use tokio::time::Instant;
use tracing::{debug, trace, warn};

use crate::config::StreamConfig;

/// Input commands to the chunker
///
/// These messages are sent to the chunker task to provide text or control
/// the chunking process.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkerInput {
    /// Text to add to the buffer and process
    Text(String),
    /// Flush the current buffer immediately
    Flush,
    /// Stop processing and flush remaining buffer
    Stop,
}

/// A text chunk ready for TTS synthesis
///
/// Chunks are emitted by the chunker when sentence boundaries are detected
/// or when size/timeout constraints require emission.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// Monotonically increasing chunk identifier
    pub id: u64,
    /// The text content to synthesize
    pub text: String,
}

impl Chunk {
    /// Create a new chunk with the given ID and text
    pub fn new(id: u64, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
        }
    }

    /// Get the number of characters in the chunk
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    /// Get the number of words in the chunk
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Check if the chunk is empty (after trimming)
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Run the chunker task
///
/// This function runs an async loop that processes `ChunkerInput` messages and emits
/// `Chunk` messages according to the configured chunking rules.
///
/// # Arguments
///
/// * `rx` - Receiver for input commands (text, flush, stop)
/// * `tx` - Sender for output chunks
/// * `config` - Shared configuration that may be updated at runtime
///
/// # Behavior
///
/// The chunker maintains an internal buffer of text. When text arrives:
/// 1. It is appended to the buffer
/// 2. The buffer is scanned for sentence boundaries
/// 3. Complete chunks are emitted
///
/// A chunk is emitted when:
/// - A sentence boundary is found after `min_chars` characters
/// - The buffer reaches `max_chars` characters
/// - A `Flush` command is received
/// - The `max_delay_ms` timeout expires
///
/// # Example
///
/// ```rust,ignore
/// let (in_tx, in_rx) = mpsc::channel(8);
/// let (out_tx, out_rx) = mpsc::channel(8);
/// let config = Arc::new(RwLock::new(StreamConfig::default()));
///
/// tokio::spawn(run_chunker(in_rx, out_tx, config));
///
/// // Send text to chunk
/// in_tx.send(ChunkerInput::Text("Hello world. ".to_string())).await;
/// in_tx.send(ChunkerInput::Text("How are you?".to_string())).await;
///
/// // Receive chunks
/// let chunk = out_rx.recv().await;
/// ```
pub async fn run_chunker(
    mut rx: mpsc::Receiver<ChunkerInput>,
    mut tx: mpsc::Sender<Chunk>,
    config: Arc<RwLock<StreamConfig>>,
) {
    let mut buffer = String::with_capacity(256);
    let mut next_id = 1u64;
    let mut buffer_started_at: Option<Instant> = None;

    trace!("Chunker started");

    loop {
        // Calculate timeout deadline before select! to avoid blocking in async context
        let timeout_sleep = async {
            if let Some(started) = buffer_started_at {
                let cfg = config.read().await;
                let deadline = started + Duration::from_millis(cfg.chunker.max_delay_ms);
                tokio::time::sleep_until(deadline).await;
            } else {
                // No buffer, sleep indefinitely (will be cancelled by recv)
                std::future::pending::<()>().await;
            }
        };

        tokio::select! {
            biased;

            Some(msg) = rx.recv() => {
                match msg {
                    ChunkerInput::Text(text) => {
                        trace!("Received text: {} chars", text.len());

                        if buffer.is_empty() {
                            buffer_started_at = Some(Instant::now());
                        }

                        buffer.push_str(&text);
                        emit_ready_chunks(&mut buffer, &mut next_id, &mut tx, &config).await;

                        if buffer.is_empty() {
                            buffer_started_at = None;
                        }
                    }
                    ChunkerInput::Flush => {
                        debug!("Flush requested, buffer has {} bytes", buffer.len());
                        flush_all(&mut buffer, &mut next_id, &mut tx).await;
                        buffer_started_at = None;
                    }
                    ChunkerInput::Stop => {
                        debug!("Stop requested, flushing {} bytes", buffer.len());
                        flush_all(&mut buffer, &mut next_id, &mut tx).await;
                        trace!("Chunker stopping");
                        break;
                    }
                }
            }
            _ = timeout_sleep => {
                // Timeout reached - emit whatever we have in the buffer
                if !buffer.is_empty() {
                    trace!("Timeout reached, emitting partial chunk");
                    let cfg = config.read().await;
                    emit_timeout_chunk(&mut buffer, &mut next_id, &mut tx, &cfg).await;

                    if buffer.is_empty() {
                        buffer_started_at = None;
                    } else {
                        buffer_started_at = Some(Instant::now());
                    }
                }
            }
            else => {
                trace!("All channels closed, chunker exiting");
                break;
            }
        }
    }

    // Final flush on exit
    if !buffer.is_empty() {
        debug!("Final flush of {} bytes on exit", buffer.len());
        flush_all(&mut buffer, &mut next_id, &mut tx).await;
    }

    trace!("Chunker stopped");
}

/// Emit all chunks that are ready from the current buffer
async fn emit_ready_chunks(
    buffer: &mut String,
    next_id: &mut u64,
    tx: &mut mpsc::Sender<Chunk>,
    config: &Arc<RwLock<StreamConfig>>,
) {
    let cfg_snapshot = { config.read().await.clone() };

    loop {
        let Some(split_idx) = find_split_index(buffer, &cfg_snapshot, false) else {
            return;
        };

        // Split the buffer at the found index
        let text = buffer[..split_idx].trim().to_string();
        buffer.replace_range(..split_idx, "");

        if !text.is_empty() {
            trace!("Emitting chunk {}: {} chars", next_id, text.len());
            if let Err(e) = tx.send(Chunk::new(*next_id, text)).await {
                warn!("Failed to send chunk: {}", e);
                return;
            }
            *next_id += 1;
        }
    }
}

/// Emit a chunk on timeout, allowing shorter chunks
async fn emit_timeout_chunk(
    buffer: &mut String,
    next_id: &mut u64,
    tx: &mut mpsc::Sender<Chunk>,
    cfg: &StreamConfig,
) {
    // On timeout, be more lenient about chunk boundaries
    let split_idx = find_split_index(buffer, cfg, true).unwrap_or(buffer.len());
    let text = buffer[..split_idx].trim().to_string();
    buffer.replace_range(..split_idx, "");

    if !text.is_empty() {
        trace!("Emitting timeout chunk {}: {} chars", next_id, text.len());
        if let Err(e) = tx.send(Chunk::new(*next_id, text)).await {
            warn!("Failed to send chunk on timeout: {}", e);
        } else {
            *next_id += 1;
        }
    }
}

/// Flush all remaining text as a single chunk
async fn flush_all(buffer: &mut String, next_id: &mut u64, tx: &mut mpsc::Sender<Chunk>) {
    let text = buffer.trim().to_string();
    buffer.clear();

    if !text.is_empty() {
        trace!("Flushing chunk {}: {} chars", next_id, text.len());
        if let Err(e) = tx.send(Chunk::new(*next_id, text)).await {
            warn!("Failed to send chunk on flush: {}", e);
        } else {
            *next_id += 1;
        }
    }
}

/// Find the index at which to split the buffer
///
/// This function scans the buffer for appropriate split points based on
/// sentence boundaries and size constraints.
///
/// # Arguments
///
/// * `buffer` - The text buffer to scan
/// * `cfg` - The chunker configuration
/// * `allow_short` - If true, allows chunks below min_chars (for timeouts)
///
/// # Returns
///
/// The byte index at which to split, or None if no split point is available
fn find_split_index(buffer: &str, cfg: &StreamConfig, allow_short: bool) -> Option<usize> {
    if buffer.is_empty() {
        return None;
    }

    let min_words = cfg.chunker.min_words.max(1);
    let min_chars = cfg.chunker.min_chars;
    let max_chars = cfg.chunker.max_chars;

    // Determine the byte index corresponding to `max_chars` characters.
    let mut total_chars = 0usize;
    let mut max_byte = buffer.len();
    for (byte_idx, ch) in buffer.char_indices() {
        total_chars += 1;
        if total_chars >= max_chars {
            max_byte = byte_idx + ch.len_utf8();
            break;
        }
    }

    // If not enough chars and not allowed short, don't split yet.
    if !allow_short && total_chars < min_chars {
        return None;
    }

    let slice = &buffer[..max_byte];

    // Prefer the *first* sentence boundary once minimum length is reached.
    let mut chars_so_far = 0usize;
    for (byte_idx, ch) in slice.char_indices() {
        chars_so_far += 1;
        if cfg.chunker.boundary_chars.contains(ch) {
            let boundary_idx = byte_idx + ch.len_utf8();
            if (allow_short || chars_so_far >= min_chars)
                && (allow_short
                    || min_words <= 1
                    || word_count(&buffer[..boundary_idx]) >= min_words)
            {
                return Some(boundary_idx);
            }
        }
    }

    // If we've reached max_chars, force a split.
    if total_chars >= max_chars {
        // Prefer whitespace for a cleaner break.
        if let Some(ws_byte) = slice.rfind(char::is_whitespace) {
            let boundary_idx = ws_byte + 1;
            if buffer.is_char_boundary(boundary_idx) {
                return Some(boundary_idx);
            }
        }
        return Some(max_byte);
    }

    None
}

/// Count the number of whitespace-separated words in text
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ChunkerConfig, GenerationConfig, StreamConfig};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio::time::{timeout, Duration};

    /// Helper to create a test configuration
    fn test_config(
        min_chars: usize,
        max_chars: usize,
        max_delay_ms: u64,
        min_words: usize,
    ) -> Arc<RwLock<StreamConfig>> {
        Arc::new(RwLock::new(StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars,
                min_words,
                max_chars,
                max_delay_ms,
                boundary_chars: ".?!".to_string(),
            },
            generation: GenerationConfig::default(),
        }))
    }

    /// Helper to receive a chunk with timeout
    async fn recv_chunk(rx: &mut mpsc::Receiver<Chunk>) -> Chunk {
        timeout(Duration::from_millis(500), rx.recv())
            .await
            .expect("timeout waiting for chunk")
            .expect("channel closed")
    }

    /// Helper to assert no chunk is received
    async fn assert_no_chunk(rx: &mut mpsc::Receiver<Chunk>) {
        let result = timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(result.is_err(), "Expected no chunk but received one");
    }

    #[test]
    fn test_chunk_new() {
        let chunk = Chunk::new(1, "Hello world");
        assert_eq!(chunk.id, 1);
        assert_eq!(chunk.text, "Hello world");
    }

    #[test]
    fn test_chunk_char_count() {
        let chunk = Chunk::new(1, "Hello");
        assert_eq!(chunk.char_count(), 5);

        let chunk_unicode = Chunk::new(1, "こんにちは"); // 5 Japanese characters
        assert_eq!(chunk_unicode.char_count(), 5);
    }

    #[test]
    fn test_chunk_word_count() {
        let chunk = Chunk::new(1, "Hello world");
        assert_eq!(chunk.word_count(), 2);

        let chunk_multi = Chunk::new(1, "One two three four");
        assert_eq!(chunk_multi.word_count(), 4);

        let chunk_empty = Chunk::new(1, "   ");
        assert_eq!(chunk_empty.word_count(), 0);
    }

    #[test]
    fn test_chunk_is_empty() {
        let chunk = Chunk::new(1, "   ");
        assert!(chunk.is_empty());

        let chunk_nonempty = Chunk::new(1, "Hello");
        assert!(!chunk_nonempty.is_empty());
    }

    #[test]
    fn test_chunker_input_equality() {
        let text1 = ChunkerInput::Text("Hello".to_string());
        let text2 = ChunkerInput::Text("Hello".to_string());
        let text3 = ChunkerInput::Text("World".to_string());

        assert_eq!(text1, text2);
        assert_ne!(text1, text3);
        assert_eq!(ChunkerInput::Flush, ChunkerInput::Flush);
        assert_eq!(ChunkerInput::Stop, ChunkerInput::Stop);
    }

    #[tokio::test]
    async fn splits_on_boundary_then_flushes_remainder() {
        let config = test_config(5, 200, 1000, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("Hello world. Next".to_string()))
            .await
            .unwrap();

        let first = recv_chunk(&mut out_rx).await;
        assert_eq!(first.id, 1);
        assert_eq!(first.text, "Hello world.");

        in_tx.send(ChunkerInput::Flush).await.unwrap();
        let second = recv_chunk(&mut out_rx).await;
        assert_eq!(second.id, 2);
        assert_eq!(second.text, "Next");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn flush_emits_short_text() {
        let config = test_config(24, 160, 1000, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("No boundaries yet".to_string()))
            .await
            .unwrap();
        in_tx.send(ChunkerInput::Flush).await.unwrap();

        let chunk = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk.text, "No boundaries yet");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn timeout_emits_short_chunk() {
        let config = test_config(100, 200, 20, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("short".to_string()))
            .await
            .unwrap();

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(80)).await;

        let chunk = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk.text, "short");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn avoids_single_word_chunks_until_more_words() {
        let config = test_config(1, 200, 1000, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("Hello.".to_string()))
            .await
            .unwrap();

        // Should not emit yet - only 1 word
        assert_no_chunk(&mut out_rx).await;

        in_tx
            .send(ChunkerInput::Text(" world.".to_string()))
            .await
            .unwrap();

        // Now should emit with both words
        let chunk = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk.text, "Hello. world.");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn handles_multiple_boundaries() {
        let config = test_config(1, 200, 1000, 1);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("First. Second. Third.".to_string()))
            .await
            .unwrap();

        let chunk1 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk1.id, 1);
        assert_eq!(chunk1.text, "First.");

        let chunk2 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk2.id, 2);
        assert_eq!(chunk2.text, "Second.");

        let chunk3 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk3.id, 3);
        assert_eq!(chunk3.text, "Third.");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn respects_max_chars_limit() {
        let config = test_config(1, 20, 1000, 1);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        // Send text longer than max_chars without boundaries
        in_tx
            .send(ChunkerInput::Text(
                "This is a very long text without boundaries".to_string(),
            ))
            .await
            .unwrap();

        // Should split at max_chars
        let chunk1 = recv_chunk(&mut out_rx).await;
        assert!(
            chunk1.text.len() <= 20,
            "Chunk too long: {}",
            chunk1.text.len()
        );

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn stop_flushes_and_ends() {
        let config = test_config(100, 200, 1000, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("Remaining text".to_string()))
            .await
            .unwrap();

        in_tx.send(ChunkerInput::Stop).await.unwrap();

        let chunk = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk.text, "Remaining text");

        // Channel should be closed now
        let result = out_rx.recv().await;
        assert!(result.is_none(), "Channel should be closed after stop");
    }

    #[tokio::test]
    async fn empty_flush_sends_nothing() {
        let config = test_config(10, 200, 1000, 2);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx.send(ChunkerInput::Flush).await.unwrap();

        // No chunk should be sent for empty buffer
        assert_no_chunk(&mut out_rx).await;

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn handles_unicode_properly() {
        // Use Japanese full-width period as boundary.
        let config = Arc::new(RwLock::new(StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 1,
                min_words: 1,
                max_chars: 200,
                max_delay_ms: 1000,
                boundary_chars: "。".to_string(),
            },
            generation: GenerationConfig::default(),
        }));
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        // Japanese text with full-width period
        in_tx
            .send(ChunkerInput::Text("こんにちは。さようなら。".to_string()))
            .await
            .unwrap();

        let chunk1 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk1.text, "こんにちは。");

        let chunk2 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk2.text, "さようなら。");

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn chunks_have_incrementing_ids() {
        let config = test_config(1, 200, 1000, 1);
        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text("One. Two. Three.".to_string()))
            .await
            .unwrap();

        let chunk1 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk1.id, 1);

        let chunk2 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk2.id, 2);

        let chunk3 = recv_chunk(&mut out_rx).await;
        assert_eq!(chunk3.id, 3);

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[tokio::test]
    async fn different_boundary_chars() {
        let config = Arc::new(RwLock::new(StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 1,
                min_words: 1,
                max_chars: 200,
                max_delay_ms: 1000,
                boundary_chars: ".?!;:\n".to_string(),
            },
            generation: GenerationConfig::default(),
        }));

        let (in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        in_tx
            .send(ChunkerInput::Text(
                "Question? Exclamation! Semi; Colon: New\nLine".to_string(),
            ))
            .await
            .unwrap();

        let chunks: Vec<_> = vec![
            recv_chunk(&mut out_rx).await,
            recv_chunk(&mut out_rx).await,
            recv_chunk(&mut out_rx).await,
            recv_chunk(&mut out_rx).await,
            recv_chunk(&mut out_rx).await,
        ];

        assert_eq!(chunks[0].text, "Question?");
        assert_eq!(chunks[1].text, "Exclamation!");
        assert_eq!(chunks[2].text, "Semi;");
        assert_eq!(chunks[3].text, "Colon:");
        assert_eq!(chunks[4].text, "New"); // \n is boundary but removed by trim

        in_tx.send(ChunkerInput::Stop).await.unwrap();
    }

    #[test]
    fn test_find_split_index_basic() {
        let config = StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 5,
                min_words: 1,
                max_chars: 100,
                max_delay_ms: 220,
                boundary_chars: ".?!".to_string(),
            },
            generation: GenerationConfig::default(),
        };

        // Should find boundary after "Hello."
        let result = find_split_index("Hello. World", &config, false);
        assert_eq!(result, Some(6)); // "Hello."

        // Should not split before min_chars
        let result = find_split_index("Hi.", &config, false);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_split_index_allow_short() {
        let config = StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 20,
                min_words: 1,
                max_chars: 100,
                max_delay_ms: 220,
                boundary_chars: ".?!".to_string(),
            },
            generation: GenerationConfig::default(),
        };

        // Without allow_short, should not split short text
        let result = find_split_index("Short.", &config, false);
        assert_eq!(result, None);

        // With allow_short, should split
        let result = find_split_index("Short.", &config, true);
        assert_eq!(result, Some(6));
    }

    #[test]
    fn test_find_split_index_max_chars() {
        let config = StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 1,
                min_words: 1,
                max_chars: 10,
                max_delay_ms: 220,
                boundary_chars: ".?!".to_string(),
            },
            generation: GenerationConfig::default(),
        };

        // Long text without boundaries should split at max_chars
        let result = find_split_index("This is very long text", &config, false);
        assert!(result.is_some());
        assert!(result.unwrap() <= 10);
    }

    #[test]
    fn test_find_split_index_empty() {
        let config = StreamConfig::default();
        let result = find_split_index("", &config, false);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_split_index_unicode_boundaries() {
        let config = StreamConfig {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig {
                min_chars: 1,
                min_words: 1,
                max_chars: 10,
                max_delay_ms: 220,
                boundary_chars: ".?!".to_string(),
            },
            generation: GenerationConfig::default(),
        };

        // Unicode text - ensure we don't split in the middle of a character
        let unicode_text = "こんにちは世界"; // 7 chars, more bytes
        let result = find_split_index(unicode_text, &config, false);

        if let Some(idx) = result {
            assert!(unicode_text.is_char_boundary(idx));
        }
    }

    #[test]
    fn test_word_count_function() {
        assert_eq!(word_count("hello world"), 2);
        assert_eq!(word_count("one two three four five"), 5);
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("   "), 0);
        assert_eq!(word_count("single"), 1);
        assert_eq!(word_count("  leading and trailing  "), 3);
    }

    #[tokio::test]
    async fn test_chunker_handles_backpressure() {
        let config = test_config(1, 200, 1000, 1);
        let (in_tx, in_rx) = mpsc::channel(2); // Small channel
        let (out_tx, mut out_rx) = mpsc::channel(2); // Small channel

        tokio::spawn(run_chunker(in_rx, out_tx, config));

        // Drain output concurrently to avoid deadlocks.
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<usize>(1);
        tokio::spawn(async move {
            let mut count = 0usize;
            while let Ok(msg) = timeout(Duration::from_millis(500), out_rx.recv()).await {
                if msg.is_none() {
                    break;
                }
                count += 1;
                if count >= 5 {
                    break;
                }
            }
            let _ = done_tx.send(count).await;
        });

        // Send multiple messages quickly
        for i in 0..5 {
            in_tx
                .send(ChunkerInput::Text(format!("Chunk {}. ", i)))
                .await
                .unwrap();
        }

        in_tx.send(ChunkerInput::Stop).await.unwrap();

        let count = timeout(Duration::from_secs(3), done_rx.recv())
            .await
            .expect("drain task timed out")
            .expect("drain task failed");

        assert_eq!(count, 5);
    }
}

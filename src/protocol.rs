//! WebSocket protocol messages for Soprano TTS client-server communication
//!
//! This module defines the message formats used for bidirectional communication
//! between clients and the Soprano TTS server over WebSocket connections.

use serde::{Deserialize, Serialize};

/// Messages sent from the client to the server
///
/// Clients use these messages to configure the TTS session, send text to synthesize,
/// and control the streaming process.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Configure the TTS session with voice, speed, and chunking parameters
    ///
    /// This message should be sent before any Text messages to establish
    /// the session configuration. If not sent, default values will be used.
    Config {
        /// Path to voice checkpoint file (optional, for future voice cloning)
        voice_path: Option<String>,
        /// Speech speed multiplier (1.0 = normal speed)
        speed: Option<f32>,
        /// Language identifier (optional, for future multilingual support)
        language_id: Option<String>,
        /// Minimum characters per chunk before emitting
        min_chars: Option<usize>,
        /// Minimum words per chunk before emitting
        min_words: Option<usize>,
        /// Maximum characters per chunk (hard limit)
        max_chars: Option<usize>,
        /// Maximum delay in milliseconds before emitting a chunk
        max_delay_ms: Option<u64>,
    },
    /// Text to synthesize
    ///
    /// The text will be buffered and chunked according to the current configuration.
    /// Chunks are emitted when sentence boundaries are detected or when
    /// timeout/size constraints are met.
    Text {
        /// The text to synthesize (UTF-8 encoded)
        text: String,
    },
    /// Flush the current buffer
    ///
    /// Forces emission of any buffered text as a chunk, even if it doesn't
    /// meet the normal criteria. Useful for ensuring prompt response at
    /// paragraph or message boundaries.
    Flush,
    /// Stop the current synthesis and clear buffers
    ///
    /// Signals that the client wants to stop the current synthesis stream.
    /// Any buffered text will be flushed, and the server will send a Done message.
    Stop,
}

impl ClientMessage {
    /// Get the voice path from a Config message
    pub fn voice_path(&self) -> Option<&str> {
        match self {
            ClientMessage::Config { voice_path, .. } => voice_path.as_deref(),
            _ => None,
        }
    }

    /// Get the speed from a Config message
    pub fn speed(&self) -> Option<f32> {
        match self {
            ClientMessage::Config { speed, .. } => *speed,
            _ => None,
        }
    }

    /// Get the language ID from a Config message
    pub fn language_id(&self) -> Option<&str> {
        match self {
            ClientMessage::Config { language_id, .. } => language_id.as_deref(),
            _ => None,
        }
    }

    /// Get the min_chars from a Config message
    pub fn min_chars(&self) -> Option<usize> {
        match self {
            ClientMessage::Config { min_chars, .. } => *min_chars,
            _ => None,
        }
    }

    /// Get the min_words from a Config message
    pub fn min_words(&self) -> Option<usize> {
        match self {
            ClientMessage::Config { min_words, .. } => *min_words,
            _ => None,
        }
    }

    /// Get the max_chars from a Config message
    pub fn max_chars(&self) -> Option<usize> {
        match self {
            ClientMessage::Config { max_chars, .. } => *max_chars,
            _ => None,
        }
    }

    /// Get the max_delay_ms from a Config message
    pub fn max_delay_ms(&self) -> Option<u64> {
        match self {
            ClientMessage::Config { max_delay_ms, .. } => *max_delay_ms,
            _ => None,
        }
    }
}

/// Messages sent from the server to the client
///
/// The server uses these messages to communicate session status, errors,
/// and stream completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Server is ready to accept text for synthesis
    ///
    /// Sent immediately after a WebSocket connection is established.
    /// Contains the unique session ID for this connection.
    Ready {
        /// Unique session identifier for this WebSocket connection
        session_id: String,
    },
    /// Error occurred during processing
    ///
    /// Sent when an error occurs that prevents normal operation.
    /// The connection may remain open for further attempts.
    Error {
        /// Human-readable error message
        message: String,
    },
    /// All synthesis complete and buffers flushed
    ///
    /// Sent after a Stop message is processed and all pending audio
    /// has been generated and sent. The client can safely close the
    /// connection or send new text.
    Done,
}

/// Header for audio frame metadata
///
/// Audio frames on the WebSocket are prefixed with this header
/// as a JSON text frame, followed by the binary audio data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioHeader {
    /// Unique identifier for this audio chunk (monotonically increasing)
    pub chunk_id: u64,
    /// Sample rate of the audio in Hz
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Audio format identifier (e.g., "pcm_s16le", "pcm_f32le")
    pub format: String,
    /// The text that was synthesized to produce this audio (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl AudioHeader {
    /// Create a new audio header with the given parameters
    pub fn new(chunk_id: u64, sample_rate: u32, channels: u16, format: impl Into<String>) -> Self {
        Self {
            chunk_id,
            sample_rate,
            channels,
            format: format.into(),
            text: None,
        }
    }

    /// Create a new audio header with text annotation
    pub fn with_text(
        chunk_id: u64,
        sample_rate: u32,
        channels: u16,
        format: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            chunk_id,
            sample_rate,
            channels,
            format: format.into(),
            text: Some(text.into()),
        }
    }

    /// Calculate the expected byte size for a given number of samples
    ///
    /// Returns the number of bytes needed to store `num_samples` samples
    /// in the format specified by this header.
    pub fn byte_size_for_samples(&self, num_samples: usize) -> usize {
        let bytes_per_sample = match self.format.as_str() {
            "pcm_s16le" | "pcm_s16be" => 2,
            "pcm_s24le" | "pcm_s24be" => 3,
            "pcm_s32le" | "pcm_s32be" | "pcm_f32le" | "pcm_f32be" => 4,
            "pcm_f64le" | "pcm_f64be" => 8,
            "pcm_u8" | "pcm_s8" => 1,
            _ => 2, // Default to 16-bit
        };
        num_samples * self.channels as usize * bytes_per_sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_message_config() {
        let json = r#"{"type":"config","voice_path":"/path/to/voice","speed":1.2,"language_id":"en","min_chars":10,"min_words":2,"max_chars":100,"max_delay_ms":500}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();

        match &msg {
            ClientMessage::Config {
                voice_path,
                speed,
                language_id,
                min_chars,
                min_words,
                max_chars,
                max_delay_ms,
            } => {
                assert_eq!(voice_path.as_deref(), Some("/path/to/voice"));
                assert_eq!(*speed, Some(1.2));
                assert_eq!(language_id.as_deref(), Some("en"));
                assert_eq!(*min_chars, Some(10));
                assert_eq!(*min_words, Some(2));
                assert_eq!(*max_chars, Some(100));
                assert_eq!(*max_delay_ms, Some(500));
            }
            _ => panic!("Expected Config message"),
        }
    }

    #[test]
    fn test_client_message_config_helpers() {
        let msg = ClientMessage::Config {
            voice_path: Some("/voice".to_string()),
            speed: Some(1.5),
            language_id: Some("en".to_string()),
            min_chars: Some(20),
            min_words: Some(3),
            max_chars: Some(200),
            max_delay_ms: Some(300),
        };

        assert_eq!(msg.voice_path(), Some("/voice"));
        assert_eq!(msg.speed(), Some(1.5));
        assert_eq!(msg.language_id(), Some("en"));
        assert_eq!(msg.min_chars(), Some(20));
        assert_eq!(msg.min_words(), Some(3));
        assert_eq!(msg.max_chars(), Some(200));
        assert_eq!(msg.max_delay_ms(), Some(300));
    }

    #[test]
    fn test_client_message_text() {
        let json = r#"{"type":"text","text":"Hello, world!"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();

        match &msg {
            ClientMessage::Text { text } => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Text message"),
        }
    }

    #[test]
    fn test_client_message_flush() {
        let json = r#"{"type":"flush"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Flush));
    }

    #[test]
    fn test_client_message_stop() {
        let json = r#"{"type":"stop"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::Stop));
    }

    #[test]
    fn test_client_message_serialization_roundtrip() {
        let original = ClientMessage::Text {
            text: "Test message".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ClientMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_server_message_ready() {
        let msg = ServerMessage::Ready {
            session_id: "sess_12345".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"ready\""));
        assert!(json.contains("\"session_id\":\"sess_12345\""));
    }

    #[test]
    fn test_server_message_error() {
        let msg = ServerMessage::Error {
            message: "Something went wrong".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"message\":\"Something went wrong\""));
    }

    #[test]
    fn test_server_message_done() {
        let msg = ServerMessage::Done;
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"done\""));
    }

    #[test]
    fn test_audio_header_new() {
        let header = AudioHeader::new(1, 32000, 1, "pcm_s16le");
        assert_eq!(header.chunk_id, 1);
        assert_eq!(header.sample_rate, 32000);
        assert_eq!(header.channels, 1);
        assert_eq!(header.format, "pcm_s16le");
        assert!(header.text.is_none());
    }

    #[test]
    fn test_audio_header_with_text() {
        let header = AudioHeader::with_text(2, 44100, 2, "pcm_f32le", "Hello world");
        assert_eq!(header.chunk_id, 2);
        assert_eq!(header.sample_rate, 44100);
        assert_eq!(header.channels, 2);
        assert_eq!(header.format, "pcm_f32le");
        assert_eq!(header.text, Some("Hello world".to_string()));
    }

    #[test]
    fn test_audio_header_byte_size() {
        let header = AudioHeader::new(1, 32000, 1, "pcm_s16le");
        assert_eq!(header.byte_size_for_samples(1000), 2000);

        let header_f32 = AudioHeader::new(1, 32000, 1, "pcm_f32le");
        assert_eq!(header_f32.byte_size_for_samples(1000), 4000);

        let header_stereo = AudioHeader::new(1, 32000, 2, "pcm_s16le");
        assert_eq!(header_stereo.byte_size_for_samples(1000), 4000);
    }

    #[test]
    fn test_audio_header_serialization() {
        let header = AudioHeader::with_text(1, 32000, 1, "pcm_s16le", "Hello");
        let json = serde_json::to_string(&header).unwrap();

        assert!(json.contains("\"chunk_id\":1"));
        assert!(json.contains("\"sample_rate\":32000"));
        assert!(json.contains("\"channels\":1"));
        assert!(json.contains("\"format\":\"pcm_s16le\""));
        assert!(json.contains("\"text\":\"Hello\""));

        // Verify roundtrip
        let deserialized: AudioHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(header, deserialized);
    }

    #[test]
    fn test_audio_header_serialization_skips_empty_text() {
        let header = AudioHeader::new(1, 32000, 1, "pcm_s16le");
        let json = serde_json::to_string(&header).unwrap();
        assert!(!json.contains("text"));
    }

    #[test]
    fn test_audio_header_default_byte_size() {
        // Test unknown format defaults to 2 bytes
        let header = AudioHeader::new(1, 32000, 1, "unknown_format");
        assert_eq!(header.byte_size_for_samples(1000), 2000);
    }

    #[test]
    fn test_server_message_equality() {
        let msg1 = ServerMessage::Ready {
            session_id: "abc".to_string(),
        };
        let msg2 = ServerMessage::Ready {
            session_id: "abc".to_string(),
        };
        let msg3 = ServerMessage::Ready {
            session_id: "def".to_string(),
        };

        assert_eq!(msg1, msg2);
        assert_ne!(msg1, msg3);
    }
}

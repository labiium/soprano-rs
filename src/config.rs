//! Configuration and CLI for Soprano TTS Server

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Command line arguments with subcommands
#[derive(Parser, Debug, Clone)]
#[command(name = "soprano-tts")]
#[command(about = "High-performance Rust TTS server and tools")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Legacy compatibility: if no subcommand provided, these are used for serve
    #[arg(long, default_value = "0.0.0.0", hide = true)]
    pub host: Option<String>,

    #[arg(long, default_value = "8080", hide = true)]
    pub port: Option<u16>,

    #[arg(long, default_value = "models", hide = true)]
    pub model_path: Option<PathBuf>,

    #[arg(long, default_value = "ekwek/Soprano-1.1-80M", hide = true)]
    pub model_id: Option<String>,

    #[arg(long, default_value = "cuda", hide = true)]
    pub device: Option<String>,

    #[arg(long, default_value = "2", hide = true)]
    pub workers: Option<usize>,

    #[arg(long, hide = true)]
    pub tts_inflight: Option<usize>,

    #[arg(long, default_value = "2", hide = true)]
    pub min_words: Option<usize>,

    #[arg(long, default_value = "false", hide = true)]
    pub include_text: Option<bool>,

    #[arg(long, default_value = "1", hide = true)]
    pub decoder_batch_size: Option<usize>,

    #[arg(long, default_value = "info", hide = true)]
    pub log: Option<String>,

    #[arg(long, default_value = "0.0", hide = true)]
    pub temperature: Option<f32>,

    #[arg(long, default_value = "0.95", hide = true)]
    pub top_p: Option<f32>,

    #[arg(long, default_value = "1.2", hide = true)]
    pub repetition_penalty: Option<f32>,

    #[arg(long, default_value = "32000", hide = true)]
    pub sample_rate: Option<u32>,

    #[arg(long, default_value = "true", hide = true)]
    pub download: Option<bool>,

    #[arg(long, hide = true)]
    pub cache_dir: Option<PathBuf>,
}

impl Cli {
    /// Get TTS inflight limit
    pub fn tts_inflight(&self) -> usize {
        // Use subcommand args if available, fall back to legacy args
        let workers = self.workers.unwrap_or(2);
        self.tts_inflight.unwrap_or_else(|| workers.max(1))
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("soprano-rs")
                .join("models")
        })
    }

    /// Convert to ServeArgs for backward compatibility
    pub fn to_serve_args(&self) -> ServeArgs {
        ServeArgs {
            host: self.host.clone().unwrap_or_else(|| "0.0.0.0".to_string()),
            port: self.port.unwrap_or(8080),
            model_path: self
                .model_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("models")),
            model_id: self
                .model_id
                .clone()
                .unwrap_or_else(|| "ekwek/Soprano-1.1-80M".to_string()),
            device: self.device.clone().unwrap_or_else(|| "cuda".to_string()),
            workers: self.workers.unwrap_or(2),
            tts_inflight: self.tts_inflight,
            min_words: self.min_words.unwrap_or(2),
            include_text: self.include_text.unwrap_or(false),
            decoder_batch_size: self.decoder_batch_size.unwrap_or(1),
            log: self.log.clone().unwrap_or_else(|| "info".to_string()),
            temperature: self.temperature.unwrap_or(0.0),
            top_p: self.top_p.unwrap_or(0.95),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.2),
            sample_rate: self.sample_rate.unwrap_or(32000),
            download: self.download.unwrap_or(true),
            cache_dir: self.cache_dir.clone(),
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Run the TTS server
    Serve(ServeArgs),
    /// Download a model from HuggingFace
    Download(DownloadArgs),
    /// List available models
    List,
    /// Show cache information
    Cache,
    /// Generate audio from text
    Generate(GenerateArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Path to model directory
    #[arg(long, default_value = "models")]
    pub model_path: PathBuf,

    /// HuggingFace model ID
    #[arg(long, default_value = "ekwek/Soprano-1.1-80M")]
    pub model_id: String,

    /// Device to use (cuda, cpu, metal)
    #[arg(long, default_value = "cuda")]
    pub device: String,

    /// Number of decoder workers
    #[arg(long, default_value_t = 2)]
    pub workers: usize,

    /// Maximum TTS requests in flight
    #[arg(long)]
    pub tts_inflight: Option<usize>,

    /// Minimum words per chunk
    #[arg(long, default_value_t = 2)]
    pub min_words: usize,

    /// Include text in audio frames
    #[arg(long, default_value_t = false)]
    pub include_text: bool,

    /// Decoder batch size
    #[arg(long, default_value_t = 1)]
    pub decoder_batch_size: usize,

    /// Log level
    #[arg(long, default_value = "info")]
    pub log: String,

    /// Temperature for generation
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-p for sampling
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.2)]
    pub repetition_penalty: f32,

    /// Sample rate for output audio
    #[arg(long, default_value_t = 32000)]
    pub sample_rate: u32,

    /// Enable model download from HuggingFace
    #[arg(long, default_value_t = true)]
    pub download: bool,

    /// Cache directory for models
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

impl ServeArgs {
    /// Get TTS inflight limit
    pub fn tts_inflight(&self) -> usize {
        self.tts_inflight.unwrap_or_else(|| self.workers.max(1))
    }

    /// Get cache directory
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("soprano-rs")
                .join("models")
        })
    }
}

#[derive(Args, Debug, Clone)]
pub struct DownloadArgs {
    /// HuggingFace model ID (e.g., ekwek/Soprano-1.1-80M)
    #[arg(value_name = "MODEL_ID")]
    pub model_id: Option<String>,

    /// Cache directory for downloaded models
    #[arg(short, long, value_name = "PATH")]
    pub cache_dir: Option<PathBuf>,

    /// Specific files to download (comma-separated)
    #[arg(short, long, value_name = "FILES")]
    pub files: Option<String>,

    /// HuggingFace endpoint URL
    #[arg(long, default_value = "https://huggingface.co")]
    pub hf_endpoint: String,

    /// Number of retry attempts for failed downloads
    #[arg(short, long, default_value_t = 3)]
    pub retries: u32,

    /// Timeout in seconds for downloads
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,

    /// Disable progress bars
    #[arg(long)]
    pub no_progress: bool,

    /// Skip checksum verification
    #[arg(long)]
    pub no_verify: bool,

    /// Clean old cached models
    #[arg(long)]
    pub cleanup: bool,

    /// Maximum age in days for cleanup (default: 30)
    #[arg(long, default_value_t = 30)]
    pub max_age: u64,

    /// Verify a downloaded model
    #[arg(long)]
    pub verify: bool,
}

#[derive(Args, Debug, Clone)]
pub struct GenerateArgs {
    /// Text to synthesize
    #[arg(short, long, group = "input")]
    pub text: Option<String>,

    /// File containing text to synthesize (one sample per line)
    #[arg(short, long, group = "input")]
    pub file: Option<PathBuf>,

    /// Output file (single sample) or directory (multiple samples)
    #[arg(short, long, default_value = "sample.wav")]
    pub output: PathBuf,

    /// Model path (directory containing model.safetensors, config.json, tokenizer.json)
    #[arg(short, long)]
    pub model_path: Option<PathBuf>,

    /// Device to use (cuda, cpu, metal)
    #[arg(short, long, default_value = "cuda")]
    pub device: String,

    /// Sample rate
    #[arg(long, default_value_t = 32000)]
    pub sample_rate: u32,

    /// Temperature for generation
    #[arg(short = 'T', long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Maximum new tokens to generate
    #[arg(long, default_value_t = 128)]
    pub max_new_tokens: usize,

    /// Top-p for sampling
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.2)]
    pub repetition_penalty: f32,

    /// Speech speed multiplier (0.5 - 2.0)
    #[arg(long, default_value_t = 1.0)]
    pub speed: f32,

    /// Number of decoder workers
    #[arg(short, long, default_value_t = 2)]
    pub workers: usize,

    /// Generate multiple variations with different seeds
    #[arg(long, default_value_t = 1)]
    pub variations: usize,

    /// Print verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

/// Streaming configuration
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// Voice path (not used in Soprano but kept for compatibility)
    pub voice_path: Option<String>,
    /// Speech speed multiplier
    pub speed: f32,
    /// Language ID (not used in Soprano - English only)
    pub language_id: Option<String>,
    /// Chunker configuration
    pub chunker: ChunkerConfig,
    /// Generation configuration
    pub generation: GenerationConfig,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            voice_path: None,
            speed: 1.0,
            language_id: None,
            chunker: ChunkerConfig::default(),
            generation: GenerationConfig::default(),
        }
    }
}

/// Chunker configuration
#[derive(Clone, Debug)]
pub struct ChunkerConfig {
    /// Minimum characters per chunk
    pub min_chars: usize,
    /// Minimum words per chunk
    pub min_words: usize,
    /// Maximum characters per chunk
    pub max_chars: usize,
    /// Maximum delay in milliseconds before emitting a chunk
    pub max_delay_ms: u64,
    /// Characters that mark sentence boundaries
    pub boundary_chars: String,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            min_chars: 24,
            min_words: 2,
            max_chars: 160,
            max_delay_ms: 220,
            boundary_chars: ".?!;:\n".to_string(),
        }
    }
}

/// Generation configuration for the LLM
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for sampling
    pub top_p: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Minimum number of new tokens before EOS can stop generation
    pub min_new_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.0,
            top_p: 0.95,
            repetition_penalty: 1.2,
            min_new_tokens: 32,
        }
    }
}

/// Model configuration
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of decoder layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Intermediate size in FFN
    pub intermediate_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 50000,
            max_position_embeddings: 2048,
        }
    }
}

/// Decoder configuration
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    /// Number of input channels
    pub num_input_channels: usize,
    /// Number of decoder layers
    pub decoder_num_layers: usize,
    /// Decoder hidden dimension
    pub decoder_dim: usize,
    /// Decoder intermediate dimension
    pub decoder_intermediate_dim: usize,
    /// Hop length for ISTFT
    pub hop_length: usize,
    /// FFT size
    pub n_fft: usize,
    /// Upscale factor
    pub upscale: usize,
    /// Depthwise kernel size
    pub dw_kernel: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            num_input_channels: 512,
            decoder_num_layers: 8,
            decoder_dim: 768,
            decoder_intermediate_dim: 2304,
            hop_length: 512,
            n_fft: 2048,
            upscale: 4,
            dw_kernel: 3,
        }
    }
}

/// Audio configuration
#[derive(Clone, Debug)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 32000,
            channels: 1,
            bits_per_sample: 16,
        }
    }
}

/// Initialize tracing with given log level
pub fn init_tracing(log_level: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

/// Load environment variables from .env file
pub fn load_dotenv() {
    let _ = dotenvy::dotenv();
}

/// Get device from string
pub fn parse_device(device_str: &str) -> candle_core::Device {
    match device_str.to_lowercase().as_str() {
        "cuda" | "gpu" => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
        "metal" | "mps" => candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
        _ => candle_core::Device::Cpu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = Cli::parse_from(["soprano-tts", "serve"]);
        assert!(matches!(cli.command, Some(Commands::Serve(_))));
    }

    #[test]
    fn test_serve_args_defaults() {
        let cli = Cli::parse_from(["soprano-tts", "serve"]);
        let args = match cli.command {
            Some(Commands::Serve(a)) => a,
            _ => panic!("expected serve command"),
        };
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 8080);
        assert_eq!(args.workers, 2);
        assert_eq!(args.min_words, 2);
    }

    #[test]
    fn test_tts_inflight() {
        let cli = Cli::parse_from(["soprano-tts", "serve"]);
        let args = match cli.command {
            Some(Commands::Serve(a)) => a,
            _ => panic!("expected serve command"),
        };
        assert_eq!(args.tts_inflight(), 2);

        let cli = Cli::parse_from(["soprano-tts", "serve", "--tts-inflight", "5"]);
        let args_with_limit = match cli.command {
            Some(Commands::Serve(a)) => a,
            _ => panic!("expected serve command"),
        };
        assert_eq!(args_with_limit.tts_inflight(), 5);
    }

    #[test]
    fn test_chunker_config_default() {
        let config = ChunkerConfig::default();
        assert_eq!(config.min_chars, 24);
        assert_eq!(config.min_words, 2);
        assert_eq!(config.max_chars, 160);
        assert_eq!(config.max_delay_ms, 220);
    }

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 512);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_p, 0.95);
        assert_eq!(config.repetition_penalty, 1.2);
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 32000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.bits_per_sample, 16);
    }
}

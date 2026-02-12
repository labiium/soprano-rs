pub mod chunker;
pub mod config;
pub mod decoder;
pub mod model;
pub mod model_loader;
pub mod normalization;
pub mod protocol;
pub mod qwen3;
pub mod server;
pub mod spectral;
pub mod splitter;
pub mod tts;

pub use config::{Cli, Commands, DownloadArgs, EngineId, GenerateArgs, ServeArgs, StreamConfig};
pub use server::AppState;
pub use tts::{
    AudioFormat, EngineInfo, GenerationMetadata, SopranoEngineConfig, SopranoTtsEngine,
    SopranoTtsEngineBuilder, TtsChunk, TtsEngine, TtsError, TtsRequest, TtsResponse,
};

// Re-export commonly used items from model_loader
#[cfg(feature = "model-download")]
pub use model_loader::{
    download_model, load_model_from_hf, load_model_from_path, DownloadConfig, ModelCache,
    ModelLoaderError,
};

pub use model_loader::{
    cuda_available, get_optimal_device, list_available_models, load_decoder_weights,
    load_model_weights, ModelFormat, ModelInfo,
};

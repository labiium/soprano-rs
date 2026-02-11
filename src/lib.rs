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

pub use config::{Cli, Commands, DownloadArgs, GenerateArgs, ServeArgs, StreamConfig};
pub use server::AppState;
pub use tts::{
    AudioFormat, EngineInfo, GenerationMetadata, SopranoEngineConfig, SopranoTtsEngine,
    SopranoTtsEngineBuilder, TtsChunk, TtsEngine, TtsError, TtsRequest, TtsResponse,
};

// Re-export commonly used items from model_loader
#[cfg(feature = "model-download")]
pub use model_loader::{
    download_model, load_model_from_hf, load_model_from_path, ModelCache,
    ModelLoaderError, DownloadConfig,
};

pub use model_loader::{
    load_model_weights, load_decoder_weights, cuda_available, get_optimal_device,
    list_available_models, ModelFormat, ModelInfo,
};

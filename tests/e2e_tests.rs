//! End-to-end tests that validate model output is real audio.
//!
//! This test synthesizes a short phrase and checks that:
//! - audio is non-empty
//! - audio has finite samples
//! - audio has non-trivial RMS/peak

mod common;

use common::{audio_validation, default_model_dir, require_model_files};

use std::time::Duration;

use soprano_tts::config::{parse_device, GenerationConfig};
use soprano_tts::tts::{SopranoEngineConfig, SopranoTtsEngine, TtsEngine, TtsRequest};

#[tokio::test]
async fn test_real_model_generates_nontrivial_audio() {
    let model_dir = default_model_dir();
    require_model_files(&model_dir);

    let device = if soprano_tts::cuda_available() {
        parse_device("cuda")
    } else {
        parse_device("cpu")
    };

    let engine = SopranoTtsEngine::new(SopranoEngineConfig {
        model_path: model_dir,
        device,
        num_workers: 1,
        sample_rate: 32000,
        generation_config: GenerationConfig {
            max_new_tokens: 256,
            min_new_tokens: 64,
            temperature: 0.0,
            top_p: 0.95,
            repetition_penalty: 1.2,
        },
        ..Default::default()
    })
    .await
    .expect("engine init failed");

    let req = TtsRequest::new("Hello, how are you? I am Mesurii.")
        .with_language("en")
        .with_speed(1.0);

    let res = tokio::time::timeout(Duration::from_secs(60), engine.synthesize(req))
        .await
        .expect("synthesis timed out")
        .expect("synthesis failed");

    audio_validation::validate_audio(&res.pcm);
    // A very conservative sanity check: this phrase should be at least ~0.6s.
    assert!(
        res.duration_secs >= 0.5,
        "audio too short ({:.3}s) - model likely stopped early",
        res.duration_secs
    );
}

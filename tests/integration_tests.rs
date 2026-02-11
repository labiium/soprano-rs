//! Integration tests for soprano-rs.
//!
//! These tests exercise the real Soprano model end-to-end. No mock engines are
//! used.

mod common;

use common::{audio_validation, default_model_dir, require_model_files};

use std::sync::Arc;
use std::time::Duration;

use soprano_tts::config::{parse_device, GenerationConfig, StreamConfig};
use soprano_tts::protocol::{ClientMessage, ServerMessage};
use soprano_tts::server::AppState;
use soprano_tts::tts::{SopranoEngineConfig, SopranoTtsEngine, TtsEngine, TtsRequest};

#[tokio::test]
async fn test_protocol_roundtrip() {
    let msg = ClientMessage::Text {
        text: "hello".to_string(),
    };
    let s = serde_json::to_string(&msg).unwrap();
    let back: ClientMessage = serde_json::from_str(&s).unwrap();
    assert_eq!(msg, back);

    let sm = ServerMessage::Ready {
        session_id: "abc".to_string(),
    };
    let s = serde_json::to_string(&sm).unwrap();
    let back: ServerMessage = serde_json::from_str(&s).unwrap();
    assert_eq!(sm, back);
}

#[tokio::test]
async fn test_engine_synthesize_real_model() {
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
        .with_speed(1.0)
        .with_generation_config(GenerationConfig {
            max_new_tokens: 256,
            min_new_tokens: 64,
            temperature: 0.0,
            top_p: 0.95,
            repetition_penalty: 1.2,
        });

    let res = tokio::time::timeout(Duration::from_secs(60), engine.synthesize(req))
        .await
        .expect("synthesis timed out")
        .expect("synthesis failed");

    assert_eq!(res.sample_rate, 32000);
    assert_eq!(res.channels, 1);
    assert!(!res.pcm.is_empty());
    audio_validation::validate_audio(&res.pcm);
}

#[tokio::test]
async fn test_server_state_construction_real_engine() {
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
        ..Default::default()
    })
    .await
    .expect("engine init failed");

    let engine: Arc<dyn TtsEngine> = Arc::new(engine);
    let cfg = StreamConfig::default();
    let _state = AppState::new(engine, cfg, 2, true);
}

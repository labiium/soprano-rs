//! Shared helpers for integration/e2e tests.
//!
//! These tests are designed to exercise the real Soprano model end-to-end.
//! No mock engines are used.

use std::path::{Path, PathBuf};

pub fn default_model_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SOPRANO_MODEL_DIR") {
        return PathBuf::from(p);
    }
    let cache = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"));
    cache
        .join("soprano-rs")
        .join("models")
        .join("ekwek--Soprano-1.1-80M")
}

pub fn require_model_files(model_dir: &Path) {
    let required = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "decoder.pth",
    ];
    for f in required {
        let p = model_dir.join(f);
        assert!(
            p.exists(),
            "missing required model file: {} (download with: soprano-tts download ekwek/Soprano-1.1-80M)",
            p.display()
        );
    }
}

pub mod audio_validation {
    pub fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s = samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32;
        s.sqrt()
    }

    pub fn peak(samples: &[f32]) -> f32 {
        samples.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    }

    pub fn validate_audio(samples: &[f32]) {
        assert!(!samples.is_empty(), "audio is empty");
        for (i, &v) in samples.iter().enumerate() {
            assert!(v.is_finite(), "non-finite sample at {i}: {v}");
        }
        let p = peak(samples);
        assert!(p <= 1.2, "peak too large: {p}");
        let r = rms(samples);
        assert!(r >= 1e-4, "rms too small (silence?): {r}");
    }
}

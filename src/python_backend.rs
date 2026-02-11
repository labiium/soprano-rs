//! Python TTS Backend
//!
//! This module provides a TTS backend that delegates to the Python
//! Soprano implementation via subprocess calls. This ensures
//! bit-for-bit compatibility with the reference implementation.

use crate::model::GenerationConfig;
use base64::{engine::general_purpose, Engine as _};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Input to the Python backend
#[derive(Serialize)]
struct PythonTtsInput {
    text: String,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
}

/// Output from the Python backend
#[derive(Deserialize)]
struct PythonTtsOutput {
    success: bool,
    audio_base64: Option<String>,
    sample_rate: Option<u32>,
    duration_seconds: Option<f32>,
    samples: Option<usize>,
    error: Option<String>,
}

/// Python-based TTS backend
#[derive(Clone)]
pub struct PythonTtsBackend {
    device: Device,
}

impl PythonTtsBackend {
    /// Create a new Python TTS backend
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// Generate audio from text using Python backend
    pub fn generate(&self, text: &str, cfg: &GenerationConfig) -> anyhow::Result<(Tensor, f32)> {
        let input = PythonTtsInput {
            text: text.to_string(),
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            repetition_penalty: cfg.repetition_penalty,
        };

        let input_json = serde_json::to_string(&input)?;

        // Call Python wrapper script
        let mut child = Command::new("python3")
            .arg("/home/emmanuel/Documents/kokoro-fast/soprano-rs/python_backend_wrapper.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Write input JSON
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(input_json.as_bytes())?;
        }

        // Wait for output
        let output = child.wait_with_output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Python backend failed: {}", stderr);
        }

        let stdout = String::from_utf8(output.stdout)?;
        let result: PythonTtsOutput = serde_json::from_str(&stdout)?;

        if !result.success {
            anyhow::bail!(
                "Python TTS failed: {}",
                result.error.unwrap_or_else(|| "Unknown error".to_string())
            );
        }

        // Decode audio
        let audio_base64 = result
            .audio_base64
            .ok_or_else(|| anyhow::anyhow!("No audio data in response"))?;
        let audio_bytes = general_purpose::STANDARD.decode(audio_base64)?;

        // Parse WAV data
        let (audio_samples, _sample_rate) = parse_wav(&audio_bytes)?;

        // Convert to tensor
        let audio_tensor = Tensor::new(audio_samples, &self.device)?;
        let duration = result.duration_seconds.unwrap_or(0.0);

        Ok((audio_tensor, duration))
    }
}

/// Parse WAV file bytes to audio samples
fn parse_wav(wav_bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    let mut cursor = Cursor::new(wav_bytes);

    // Read WAV header
    let mut riff_header = [0u8; 4];
    cursor.read_exact(&mut riff_header)?;
    if &riff_header != b"RIFF" {
        anyhow::bail!("Not a valid WAV file: missing RIFF header");
    }

    // Skip chunk size
    cursor.read_u32::<LittleEndian>()?;

    // Read WAVE header
    let mut wave_header = [0u8; 4];
    cursor.read_exact(&mut wave_header)?;
    if &wave_header != b"WAVE" {
        anyhow::bail!("Not a valid WAV file: missing WAVE header");
    }

    // Find fmt chunk
    let mut chunk_id = [0u8; 4];
    cursor.read_exact(&mut chunk_id)?;
    while &chunk_id != b"fmt " {
        let chunk_size = cursor.read_u32::<LittleEndian>()?;
        cursor.set_position(cursor.position() + chunk_size as u64);
        cursor.read_exact(&mut chunk_id)?;
    }

    // Read fmt chunk
    let _fmt_chunk_size = cursor.read_u32::<LittleEndian>()?;
    let _audio_format = cursor.read_u16::<LittleEndian>()?;
    let num_channels = cursor.read_u16::<LittleEndian>()?;
    let sample_rate = cursor.read_u32::<LittleEndian>()?;
    let _byte_rate = cursor.read_u32::<LittleEndian>()?;
    let _block_align = cursor.read_u16::<LittleEndian>()?;
    let bits_per_sample = cursor.read_u16::<LittleEndian>()?;

    // Find data chunk
    cursor.read_exact(&mut chunk_id)?;
    while &chunk_id != b"data" {
        let chunk_size = cursor.read_u32::<LittleEndian>()?;
        cursor.set_position(cursor.position() + chunk_size as u64);
        cursor.read_exact(&mut chunk_id)?;
    }

    // Read data chunk
    let data_size = cursor.read_u32::<LittleEndian>()?;
    let num_samples = data_size / (bits_per_sample / 8) as u32;

    // Read samples
    let mut samples = Vec::with_capacity(num_samples as usize);

    if bits_per_sample == 16 {
        for _ in 0..num_samples {
            let sample_i16 = cursor.read_i16::<LittleEndian>()?;
            // Convert to float in range [-1, 1]
            let sample_f32 = sample_i16 as f32 / 32768.0;
            samples.push(sample_f32);
        }
    } else {
        anyhow::bail!("Unsupported bits per sample: {}", bits_per_sample);
    }

    // Convert mono to expected format (already mono, just ensure correct shape)
    if num_channels != 1 {
        // Average channels if stereo
        let mut mono_samples = Vec::with_capacity(samples.len() / num_channels as usize);
        for chunk in samples.chunks(num_channels as usize) {
            let avg = chunk.iter().sum::<f32>() / num_channels as f32;
            mono_samples.push(avg);
        }
        samples = mono_samples;
    }

    Ok((samples, sample_rate))
}

/// Result from Python backend generation
#[derive(Clone, Debug)]
pub struct PythonGenerationResult {
    /// Audio samples
    pub audio: Tensor,
    /// Duration in seconds
    pub duration: f32,
    /// Sample rate
    pub sample_rate: u32,
}

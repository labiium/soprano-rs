//! Simple TTS Usage Example
//!
//! This example demonstrates the simplest way to use the Soprano TTS library
//! programmatically in Rust code.
//!
//! Usage:
//!   cargo run --example simple_tts -- --text "Hello, world!"
//!
//! Or use default text:
//!   cargo run --example simple_tts

use clap::Parser;
use soprano::{
    config::{parse_device, GenerationConfig},
    tts::{SopranoTtsEngineBuilder, TtsEngine, TtsRequest},
};
use std::path::PathBuf;
use std::time::Instant;

/// Simple TTS example
#[derive(Parser, Debug)]
#[command(name = "simple_tts")]
#[command(about = "Simple TTS usage example")]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello, this is Soprano text to speech!")]
    text: String,

    /// Model path
    #[arg(short, long, default_value = "models")]
    model_path: PathBuf,

    /// Device (cpu, cuda, metal)
    #[arg(short, long, default_value = "cuda")]
    device: String,

    /// Output file (WAV format)
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Speech speed (0.5 - 2.0)
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Number of decoder workers
    #[arg(short, long, default_value_t = 2)]
    workers: usize,

    /// Temperature for generation
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Top-p for sampling
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Soprano TTS Simple Example");
    println!("==========================");
    println!("Text: {}", args.text);
    println!("Model path: {:?}", args.model_path);
    println!("Device: {}", args.device);
    println!("Workers: {}", args.workers);
    println!();

    // Parse device
    let device = parse_device(&args.device);
    println!("Using device: {:?}", device);

    // Check if model exists
    if !args.model_path.exists() {
        println!("Warning: Model path does not exist: {:?}", args.model_path);
        println!("Please download a model first using: cargo run --bin soprano-download");
        return Ok(());
    }

    // Create generation config
    let gen_config = GenerationConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        ..Default::default()
    };

    // Build TTS engine
    println!("Initializing TTS engine...");
    let init_start = Instant::now();

    let engine = SopranoTtsEngineBuilder::new(&args.model_path)
        .with_device(device)
        .with_workers(args.workers)
        .with_generation_config(gen_config)
        .build()
        .await?;

    println!("Engine initialized in {:?}", init_start.elapsed());

    // Get engine info
    let info = engine.info();
    println!("Engine info:");
    println!("  Name: {}", info.name);
    println!("  Version: {}", info.version);
    println!("  Sample rate: {} Hz", info.sample_rate);
    println!("  Supports streaming: {}", info.supports_streaming);
    println!();

    // Create TTS request
    let request = TtsRequest::new(&args.text)
        .with_speed(args.speed)
        .with_language("en");

    // Synthesize
    println!("Synthesizing...");
    let synthesis_start = Instant::now();

    let response = engine.synthesize(request).await?;

    let synthesis_duration = synthesis_start.elapsed();
    println!("Synthesis completed in {:?}", synthesis_duration);

    // Print response info
    println!("\nResponse info:");
    println!("  Sample rate: {} Hz", response.sample_rate);
    println!("  Channels: {}", response.channels);
    println!("  Duration: {:.2}s", response.duration_secs);
    println!("  Samples: {}", response.num_samples);
    println!("  Format: {}", response.format);

    // Print metadata
    println!("\nGeneration metadata:");
    println!("  Tokens generated: {}", response.metadata.tokens_generated);
    println!(
        "  Sentences processed: {}",
        response.metadata.sentences_processed
    );
    println!(
        "  Processing time: {}ms",
        response.metadata.processing_time_ms
    );
    println!("  LLM time: {}ms", response.metadata.llm_time_ms);
    println!("  Decoder time: {}ms", response.metadata.decoder_time_ms);
    println!("  Finish reason: {}", response.metadata.finish_reason);

    // Calculate RTF (real-time factor)
    if synthesis_duration.as_secs_f32() > 0.0 {
        let rtf = synthesis_duration.as_secs_f32() / response.duration_secs;
        println!("\nReal-time factor: {:.3} (lower is better)", rtf);
        println!("  - RTF < 1.0: Faster than real-time");
        println!("  - RTF = 1.0: Real-time");
        println!("  - RTF > 1.0: Slower than real-time");
    }

    // Save audio as WAV
    println!("\nSaving audio to: {:?}", args.output);
    save_wav(&response.pcm, response.sample_rate, &args.output)?;
    println!("Audio saved successfully!");

    // Also save raw PCM for comparison
    let pcm_path = args.output.with_extension("pcm");
    println!("Also saving raw PCM to: {:?}", pcm_path);
    std::fs::write(&pcm_path, pcm_to_bytes(&response.pcm))?;

    Ok(())
}

/// Convert f32 PCM samples to i16 bytes
fn pcm_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&i16_sample.to_le_bytes());
    }
    bytes
}

/// Save audio as WAV file
fn save_wav(
    samples: &[f32],
    sample_rate: u32,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = channels * (bits_per_sample / 8);
    let data_size = samples.len() * (bits_per_sample as usize / 8);

    let mut file = std::fs::File::create(path)?;

    // RIFF chunk
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size as u32).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt subchunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    file.write_all(b"data")?;
    file.write_all(&(data_size as u32).to_le_bytes())?;

    // Write samples
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        file.write_all(&i16_sample.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}

/// Example: Batch processing
#[allow(dead_code)]
async fn batch_example(
    engine: &soprano::tts::SopranoTtsEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    let texts = vec![
        "First sentence to synthesize.",
        "Second sentence for batch processing.",
        "Third and final sentence.",
    ];

    println!("\nBatch processing {} texts...", texts.len());

    let requests: Vec<_> = texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| TtsRequest::new(text).with_id(i as u64 + 1).with_speed(1.0))
        .collect();

    let batch_start = Instant::now();
    let results = engine.synthesize_batch(requests).await;
    let batch_duration = batch_start.elapsed();

    println!("Batch completed in {:?}", batch_duration);

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(response) => {
                println!(
                    "  [{}] Success: {:.2}s audio",
                    i + 1,
                    response.duration_secs
                );
            }
            Err(e) => {
                println!("  [{}] Error: {}", i + 1, e);
            }
        }
    }

    Ok(())
}

/// Example: Streaming synthesis
#[allow(dead_code)]
async fn streaming_example(
    engine: &soprano::tts::SopranoTtsEngine,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::time::{timeout, Duration};

    println!("\nStreaming synthesis example...");

    let request = TtsRequest::new(text).with_streaming(true);

    let mut rx = engine.synthesize_streaming(request).await?;

    let mut chunk_count = 0;
    let mut total_samples = 0;

    while let Ok(Some(result)) = timeout(Duration::from_secs(30), rx.recv()).await {
        match result {
            Ok(chunk) => {
                chunk_count += 1;
                total_samples += chunk.pcm.len();
                println!(
                    "  Chunk {}: {} samples (final: {})",
                    chunk.sequence,
                    chunk.pcm.len(),
                    chunk.is_final
                );

                if chunk.is_final {
                    break;
                }
            }
            Err(e) => {
                println!("  Error: {}", e);
                break;
            }
        }
    }

    println!(
        "Stream complete: {} chunks, {} total samples",
        chunk_count, total_samples
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_to_bytes() {
        let samples = vec![0.0f32, 1.0, -1.0, 0.5, -0.5];
        let bytes = pcm_to_bytes(&samples);

        assert_eq!(bytes.len(), samples.len() * 2);

        // Check max value
        let max = i16::from_le_bytes([bytes[2], bytes[3]]);
        assert_eq!(max, 32767);

        // Check min value
        let min = i16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(min, -32767);
    }

    #[test]
    fn test_save_wav() {
        let samples = vec![0.0f32; 1000];
        let temp_path = std::env::temp_dir().join("test.wav");

        save_wav(&samples, 32000, &temp_path).unwrap();

        // Verify file exists and has correct header
        let content = std::fs::read(&temp_path).unwrap();
        assert!(content.starts_with(b"RIFF"));
        assert!(content[8..12].eq(b"WAVE"));

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}

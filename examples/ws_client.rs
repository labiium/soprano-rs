//! WebSocket Client Example for Soprano TTS
//!
//! This example demonstrates how to connect to the Soprano TTS server
//! via WebSocket and stream text-to-speech audio.
//!
//! Usage:
//!   cargo run --example ws_client -- --url ws://localhost:8080/v1/stream
//!
//! Or with options:
//!   cargo run --example ws_client -- \
//!     --url ws://localhost:8080/v1/stream \
//!     --text "Hello, world!" \
//!     --output output.raw

use clap::Parser;
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Instant;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// WebSocket client for Soprano TTS
#[derive(Parser, Debug)]
#[command(name = "ws_client")]
#[command(about = "WebSocket client for Soprano TTS")]
struct Args {
    /// WebSocket URL
    #[arg(short, long, default_value = "ws://localhost:8080/v1/stream")]
    url: String,

    /// Text to synthesize
    #[arg(short, long, default_value = "Hello, this is a test of the Soprano text to speech system!")]
    text: String,

    /// Output file for audio (raw PCM)
    #[arg(short, long, default_value = "output.raw")]
    output: String,

    /// Speech speed multiplier
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Minimum words per chunk
    #[arg(long, default_value_t = 2)]
    min_words: usize,

    /// Include text annotation in audio frames
    #[arg(long)]
    include_text: bool,

    /// Convert to WAV format
    #[arg(long)]
    wav: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Audio frame header parsed from binary message
#[derive(Debug)]
struct AudioFrame {
    chunk_id: u64,
    sample_rate: u32,
    channels: u16,
    #[allow(dead_code)]
    format: String,
    #[allow(dead_code)]
    text: Option<String>,
    samples: Vec<i16>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.verbose {
        println!("Connecting to: {}", args.url);
        println!("Text: {}", args.text);
    }

    // Connect to WebSocket
    let start_time = Instant::now();
    let (mut ws, response) = connect_async(&args.url).await?;

    if args.verbose {
        println!("Connected in {:?}", start_time.elapsed());
        println!("Response: {:?}", response);
    }

    // Wait for ready message
    let ready_msg = ws
        .next()
        .await
        .ok_or("Connection closed before ready")??;

    let _session_id = match ready_msg {
        Message::Text(text) => {
            let msg: serde_json::Value = serde_json::from_str(&text)?;
            if msg["type"] == "ready" {
                let id = msg["session_id"].as_str().unwrap_or("unknown").to_string();
                println!("Session ID: {}", id);
                id
            } else {
                return Err(format!("Unexpected message: {}", text).into());
            }
        }
        _ => return Err("Expected text message for ready".into()),
    };

    // Send configuration
    let config = json!({
        "type": "config",
        "speed": args.speed,
        "min_words": args.min_words,
    });

    if args.verbose {
        println!("Sending config: {}", config);
    }

    ws.send(Message::Text(config.to_string())).await?;

    // Send text
    let text_msg = json!({
        "type": "text",
        "text": args.text,
    });

    ws.send(Message::Text(text_msg.to_string())).await?;

    // Send flush
    ws.send(Message::Text(json!({"type": "flush"}).to_string())).await?;

    // Collect audio frames
    let mut frames: Vec<AudioFrame> = Vec::new();
    let mut total_samples = 0usize;
    let synthesis_start = Instant::now();
    let mut first_chunk_time: Option<Instant> = None;

    while let Some(msg) = ws.next().await {
        match msg? {
            Message::Binary(data) => {
                if first_chunk_time.is_none() {
                    first_chunk_time = Some(Instant::now());
                    println!(
                        "First chunk latency: {:?}",
                        first_chunk_time.unwrap().duration_since(synthesis_start)
                    );
                }

                match parse_audio_frame(&data, args.include_text) {
                    Ok(frame) => {
                        if args.verbose {
                            println!(
                                "Received chunk {}: {} samples ({} bytes)",
                                frame.chunk_id,
                                frame.samples.len(),
                                data.len()
                            );
                        }
                        total_samples += frame.samples.len();
                        frames.push(frame);
                    }
                    Err(e) => {
                        eprintln!("Error parsing audio frame: {}", e);
                    }
                }
            }
            Message::Text(text) => {
                let msg: serde_json::Value = serde_json::from_str(&text)?;
                match msg["type"].as_str() {
                    Some("done") => {
                        println!("Synthesis complete");
                        break;
                    }
                    Some("error") => {
                        eprintln!("Server error: {}", msg["message"].as_str().unwrap_or("unknown"));
                        return Err("Server error".into());
                    }
                    _ => {
                        if args.verbose {
                            println!("Server message: {}", text);
                        }
                    }
                }
            }
            Message::Close(_) => {
                println!("Connection closed");
                break;
            }
            _ => {}
        }
    }

    let total_duration = synthesis_start.elapsed();
    println!("Total synthesis time: {:?}", total_duration);
    println!("Total frames: {}", frames.len());
    println!("Total samples: {}", total_samples);

    if !frames.is_empty() {
        let sample_rate = frames[0].sample_rate as f32;
        let audio_duration_secs = total_samples as f32 / sample_rate;
        println!("Audio duration: {:.2}s", audio_duration_secs);

        if total_duration.as_secs_f32() > 0.0 {
            let rtf = total_duration.as_secs_f32() / audio_duration_secs;
            println!("Real-time factor: {:.3}", rtf);
        }

        // Save audio
        if args.wav {
            save_as_wav(&frames, &args.output).await?;
        } else {
            save_raw_pcm(&frames, &args.output).await?;
        }

        println!("Audio saved to: {}", args.output);
    }

    // Send stop message and close
    let _ = ws.send(Message::Text(json!({"type": "stop"}).to_string())).await;
    let _ = ws.close(None).await;

    Ok(())
}

/// Parse binary audio frame
fn parse_audio_frame(data: &[u8], extract_text: bool) -> Result<AudioFrame, Box<dyn std::error::Error>> {
    if data.len() < 4 {
        return Err("Data too short".into());
    }

    // Parse header length (big-endian u32)
    let header_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

    if data.len() < 4 + header_len {
        return Err("Incomplete header".into());
    }

    // Parse JSON header
    let header_json: serde_json::Value = serde_json::from_slice(&data[4..4 + header_len])?;

    let chunk_id = header_json["chunk_id"].as_u64().unwrap_or(0);
    let sample_rate = header_json["sample_rate"].as_u64().unwrap_or(32000) as u32;
    let channels = header_json["channels"].as_u64().unwrap_or(1) as u16;
    let format = header_json["format"]
        .as_str()
        .unwrap_or("pcm_s16le")
        .to_string();

    let text = if extract_text {
        header_json["text"].as_str().map(|s| s.to_string())
    } else {
        None
    };

    // Parse PCM data (i16 little-endian)
    let pcm_data = &data[4 + header_len..];
    let mut samples = Vec::with_capacity(pcm_data.len() / 2);

    for i in (0..pcm_data.len()).step_by(2) {
        if i + 1 < pcm_data.len() {
            let sample = i16::from_le_bytes([pcm_data[i], pcm_data[i + 1]]);
            samples.push(sample);
        }
    }

    Ok(AudioFrame {
        chunk_id,
        sample_rate,
        channels,
        format,
        text,
        samples,
    })
}

/// Save audio frames as raw PCM
async fn save_raw_pcm(frames: &[AudioFrame], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path).await?;

    for frame in frames {
        for sample in &frame.samples {
            file.write_all(&sample.to_le_bytes()).await?;
        }
    }

    file.flush().await?;
    Ok(())
}

/// Save audio frames as WAV file
async fn save_as_wav(frames: &[AudioFrame], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    if frames.is_empty() {
        return Err("No frames to save".into());
    }

    let sample_rate = frames[0].sample_rate;
    let channels = frames[0].channels;
    let bits_per_sample = 16u16;

    // Calculate total samples
    let total_samples: usize = frames.iter().map(|f| f.samples.len()).sum();
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = channels * (bits_per_sample / 8);
    let data_size = total_samples * (bits_per_sample as usize / 8);

    // Create file
    let mut file = std::fs::File::create(path)?;

    // Write WAV header
    // RIFF chunk
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size as u32).to_le_bytes())?; // File size - 8
    file.write_all(b"WAVE")?;

    // fmt subchunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk size
    file.write_all(&1u16.to_le_bytes())?; // Audio format (PCM)
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    file.write_all(b"data")?;
    file.write_all(&(data_size as u32).to_le_bytes())?;

    // Write samples
    for frame in frames {
        for sample in &frame.samples {
            file.write_all(&sample.to_le_bytes())?;
        }
    }

    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_audio_frame() {
        // Create test frame
        let header = json!({
            "chunk_id": 1,
            "sample_rate": 32000,
            "channels": 1,
            "format": "pcm_s16le"
        });
        let header_bytes = header.to_string().as_bytes().to_vec();
        let header_len = header_bytes.len() as u32;

        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_be_bytes());
        data.extend_from_slice(&header_bytes);
        // Add 2 samples (4 bytes)
        data.extend_from_slice(&1000i16.to_le_bytes());
        data.extend_from_slice(&(-1000i16).to_le_bytes());

        let frame = parse_audio_frame(&data, false).unwrap();
        assert_eq!(frame.chunk_id, 1);
        assert_eq!(frame.sample_rate, 32000);
        assert_eq!(frame.samples.len(), 2);
    }
}

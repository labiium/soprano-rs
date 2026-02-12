# Soprano-RS

High-performance Rust TTS server and CLI for the Soprano model, built on Candle.

## Status

- Engine support: `soprano` only
- Main interface: CLI + WebSocket streaming API
- Audio output: PCM/WAV generation via CLI, binary PCM frames over WebSocket

## Features

- Streaming TTS over WebSocket (`/v1/stream`)
- Health endpoint (`/healthz`)
- Model download/list/cache utilities
- Configurable generation and chunking parameters
- CUDA execution path with optional flash-attn feature

## Requirements

- Rust + Cargo
- Linux/macOS
- Optional NVIDIA GPU + CUDA for acceleration
- Soprano model files (local or downloaded)

Default model ID:

- `ekwek/Soprano-1.1-80M`

## Build

```bash
cargo build --release
```

## Install

Install the latest binary directly from GitHub:

```bash
cargo install --git https://github.com/labiium/soprano-rs --locked
```

Then run:

```bash
soprano --help
```

You can also run from source without installing:

```bash
cargo run -- --help
```

## CLI

```bash
soprano [COMMAND]
```

Commands:

- `serve` - Run TTS server
- `download` - Download model from HuggingFace
- `list` - List available models
- `cache` - Show cache information
- `generate` - Generate audio from text/file

If no command is provided, it defaults to `serve`.

### Examples

```bash
# List models
soprano list

# Download default model
soprano download ekwek/Soprano-1.1-80M

# Start server
soprano serve --host 0.0.0.0 --port 8080

# Generate one WAV file
soprano generate --text "Hello from soprano-rs" --output sample.wav

# Generate from file (one line per sample)
soprano generate --file input.txt --output ./output
```

## Library usage

You can use this crate directly as a Rust library for one-shot or streaming synthesis.

Add dependencies:

```toml
[dependencies]
soprano = { git = "https://github.com/labiium/soprano-rs" }
tokio = { version = "1", features = ["full"] }
hound = "3.5"
```

### One-shot synthesis (text -> WAV)

```rust
use candle_core::Device;
use hound::{SampleFormat, WavSpec, WavWriter};
use soprano::{SopranoTtsEngineBuilder, TtsEngine, TtsRequest};

fn select_device(name: &str) -> Device {
    match name.to_lowercase().as_str() {
        "cuda" | "gpu" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        "metal" | "mps" => Device::new_metal(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = select_device("cuda");

    let engine = SopranoTtsEngineBuilder::new("models")
        .with_device(device)
        .with_workers(2)
        .build()
        .await?;

    let req = TtsRequest::new("Hello from the soprano library API");
    let resp = engine.synthesize(req).await?;

    write_wav("sample.wav", &resp.pcm, resp.sample_rate)?;
    println!("Wrote sample.wav ({} samples @ {} Hz)", resp.num_samples, resp.sample_rate);
    Ok(())
}
```

### Streaming synthesis (chunked output)

```rust
use candle_core::Device;
use hound::{SampleFormat, WavSpec, WavWriter};
use soprano::{SopranoTtsEngineBuilder, TtsEngine, TtsRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = SopranoTtsEngineBuilder::new("models")
        .with_device(Device::new_cuda(0).unwrap_or(Device::Cpu))
        .build()
        .await?;

    let mut rx = engine
        .synthesize_streaming(TtsRequest::new(
            "This is a streaming synthesis example using the soprano library API.",
        ))
        .await?;

    let mut all_samples = Vec::new();
    while let Some(chunk_result) = rx.recv().await {
        let chunk = chunk_result?;
        println!("chunk {}: {} samples", chunk.sequence, chunk.pcm.len());
        all_samples.extend_from_slice(&chunk.pcm);
        if chunk.is_final {
            break;
        }
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: 32000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create("stream.wav", spec)?;
    for s in all_samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
```

## Server API

### Health

- `GET /healthz`
- Response body: `ok`

### Streaming endpoint

- `GET /v1/stream` (WebSocket upgrade)

Client JSON messages:

- `{"type":"config", ...}`
- `{"type":"text", "text":"..."}`
- `{"type":"flush"}`
- `{"type":"stop"}`

Server JSON messages:

- `{"type":"ready", "session_id":"..."}`
- `{"type":"error", "message":"..."}`
- `{"type":"done"}`

Binary audio frame format:

1. 4-byte big-endian header length
2. JSON header bytes (`chunk_id`, `sample_rate`, `channels`, `format`, optional `text`)
3. PCM payload bytes (i16 little-endian)

## Configuration

You can configure runtime via CLI flags and environment variables (see `.env.example`).

Common server options:

- `--host`, `--port`
- `--model-path`, `--model-id`, `--download`, `--cache-dir`
- `--device` (`cuda`, `cpu`, `metal`)
- `--workers`, `--tts-inflight`
- `--temperature`, `--top-p`, `--repetition-penalty`
- `--sample-rate`

## License

Licensed under Apache License 2.0. See `LICENSE`.

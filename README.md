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
- **Automatic device detection** - uses best available accelerator (CUDA/Metal/CPU)
- Cross-platform: Linux, macOS, Windows

## Requirements

- Rust + Cargo
- Linux/macOS/Windows
- Optional: NVIDIA GPU (CUDA) or Apple Silicon (Metal) for acceleration
- Soprano model files (local or downloaded)

Default model ID:

- `ekwek/Soprano-1.1-80M`

## Build

One command builds for all platforms with automatic accelerator detection:

```bash
cargo build --release
```

The binary uses **dynamic-loading** for accelerators:
- **macOS**: Includes Metal support (auto-detected at runtime, requires macOS 10.14+)
- **Linux/Windows**: CUDA support auto-detected at runtime if NVIDIA drivers installed
- **All platforms**: Falls back to CPU if no GPU available

No CUDA toolkit required at build time!

### Pre-built Binaries

Download from [GitHub Releases](https://github.com/labiium/soprano-rs/releases):

| Platform | Binary | Accelerators |
|----------|--------|--------------|
| macOS | `soprano-macos.tar.gz` | Metal, CPU |
| Linux | `soprano-linux.tar.gz` | CUDA*, CPU |
| Windows | `soprano-windows.zip` | CUDA*, CPU |

\* CUDA support requires NVIDIA drivers installed. Binary uses dynamic-loading to detect at runtime.

All binaries auto-detect the best available accelerator.
cargo build --release --features cuda
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
- `devices` - Show available compute devices
- `generate` - Generate audio from text/file

If no command is provided, it defaults to `serve`.

### Examples

```bash
# List models
soprano list

# Show available compute devices
soprano devices

# Download default model
soprano download ekwek/Soprano-1.1-80M

# Start server (auto-detects best device)
soprano serve --host 0.0.0.0 --port 8080

# Force specific device
soprano serve --device cuda      # NVIDIA GPU
soprano serve --device metal     # Apple GPU (macOS)
soprano serve --device cpu       # CPU only

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
use soprano::{SopranoTtsEngineBuilder, TtsEngine, TtsRequest, auto_select_device};

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
    // Auto-select the best available device
    let device = auto_select_device();
    println!("Using device: {:?}", device);

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
use soprano::{SopranoTtsEngineBuilder, TtsEngine, TtsRequest, auto_select_device};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = SopranoTtsEngineBuilder::new("models")
        .with_device(auto_select_device())
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

### Device Selection

The `--device` option supports:

- `auto` (default) - Automatically detect and use the best available accelerator:
  - macOS: Metal (Apple Silicon/AMD) → CPU
  - Linux/Windows: CUDA (NVIDIA) → CPU
- `cuda` - Force NVIDIA CUDA GPU
- `metal` - Force Apple Metal (macOS only)
- `cpu` - Force CPU only

### Common Server Options

- `--host`, `--port`
- `--model-path`, `--model-id`, `--download`, `--cache-dir`
- `--device` (`auto`, `cuda`, `metal`, `cpu`)
- `--workers`, `--tts-inflight`
- `--temperature`, `--top-p`, `--repetition-penalty`
- `--sample-rate`

## License

Licensed under Apache License 2.0. See `LICENSE`.

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
cargo run -- list

# Download default model
cargo run -- download ekwek/Soprano-1.1-80M

# Start server
cargo run -- serve --host 0.0.0.0 --port 8080

# Generate one WAV file
cargo run -- generate --text "Hello from soprano-rs" --output sample.wav

# Generate from file (one line per sample)
cargo run -- generate --file input.txt --output ./output
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

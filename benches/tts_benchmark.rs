//! Criterion benchmarks for Soprano TTS
//!
//! Run with: cargo bench

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::runtime::Runtime;

use soprano::chunker::{run_chunker, ChunkerInput};
use soprano::config::{ChunkerConfig, GenerationConfig, StreamConfig};
use soprano::normalization::clean_text;
use soprano::splitter::split_and_recombine_text;

/// Helper to create a Tokio runtime for async benchmarks
fn tokio_runtime() -> Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

/// Create a standard test config
fn test_config() -> StreamConfig {
    StreamConfig {
        voice_path: None,
        speed: 1.0,
        language_id: None,
        chunker: ChunkerConfig {
            min_chars: 24,
            min_words: 2,
            max_chars: 160,
            max_delay_ms: 220,
            boundary_chars: ".?!;:\n".to_string(),
        },
        generation: GenerationConfig::default(),
    }
}

/// Benchmark text normalization
fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    let texts = vec![
        ("short", "Hello world!"),
        (
            "medium",
            "Dr. Smith has 5 patients today. The meeting is at 3:00 PM!",
        ),
        (
            "long",
            "Dr. Smith has 5 patients today. The meeting is at 3:00 PM. \
                  Mr. Johnson works at IBM and uses CPU and GPU for processing. \
                  The price is $100.50 and he has 1st place in the competition.",
        ),
        ("with_numbers", "1 2 3 4 5 6 7 8 9 10. 100 200 300 400 500."),
        (
            "with_abbreviations",
            "Dr. Mr. Mrs. Ms. Prof. Sr. Jr. St. Ave. Blvd. Rd. \
                               USA UK EU UN NATO FBI CIA NASA MIT IBM CPU GPU API CLI",
        ),
    ];

    for (name, text) in texts {
        group.bench_with_input(BenchmarkId::new("clean_text", name), text, |b, text| {
            b.iter(|| clean_text(black_box(text)));
        });
    }

    group.finish();
}

/// Benchmark text splitting
fn bench_text_splitter(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_splitter");

    let texts = vec![
        ("short", "Hello world. This is a test."),
        (
            "medium",
            "First sentence. Second sentence. Third sentence. \
                    Fourth sentence. Fifth sentence.",
        ),
        (
            "long",
            "The quick brown fox jumps over the lazy dog. \
                  This is a longer paragraph with multiple sentences. \
                  Each sentence should be processed correctly. \
                  The text splitter should handle this well. \
                  We want to test performance with various lengths.",
        ),
    ];

    let desired_lengths = vec![50, 100, 200];

    for (name, text) in texts {
        for &desired_len in &desired_lengths {
            let bench_name = format!("{}/desired_{}", name, desired_len);
            group.bench_function(bench_name, |b| {
                b.iter(|| {
                    split_and_recombine_text(
                        black_box(text),
                        black_box(desired_len),
                        black_box(desired_len * 2),
                    )
                });
            });
        }
    }

    group.finish();
}

/// Benchmark chunker
fn bench_chunker(c: &mut Criterion) {
    let rt = tokio_runtime();

    let mut group = c.benchmark_group("chunker");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("short_text", |b| {
        b.to_async(&rt).iter(|| async {
            let config = Arc::new(tokio::sync::RwLock::new(test_config()));
            let (in_tx, in_rx) = tokio::sync::mpsc::channel(8);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(8);

            tokio::spawn(run_chunker(in_rx, out_tx, config));

            in_tx
                .send(ChunkerInput::Text("Hello world. ".to_string()))
                .await
                .unwrap();
            in_tx
                .send(ChunkerInput::Text("How are you?".to_string()))
                .await
                .unwrap();
            in_tx.send(ChunkerInput::Stop).await.unwrap();

            while out_rx.recv().await.is_some() {}
        });
    });

    group.bench_function("medium_text", |b| {
        b.to_async(&rt).iter(|| async {
            let config = Arc::new(tokio::sync::RwLock::new(test_config()));
            let (in_tx, in_rx) = tokio::sync::mpsc::channel(8);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(8);

            tokio::spawn(run_chunker(in_rx, out_tx, config));

            let text = "First sentence. Second sentence. Third sentence. \
                       Fourth sentence. Fifth sentence.";
            in_tx
                .send(ChunkerInput::Text(text.to_string()))
                .await
                .unwrap();
            in_tx.send(ChunkerInput::Stop).await.unwrap();

            while out_rx.recv().await.is_some() {}
        });
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("high_throughput", |b| {
        b.to_async(&rt).iter(|| async {
            let config = Arc::new(tokio::sync::RwLock::new(test_config()));
            let (in_tx, in_rx) = tokio::sync::mpsc::channel(100);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(100);

            tokio::spawn(run_chunker(in_rx, out_tx, config));

            for i in 0..100 {
                in_tx
                    .send(ChunkerInput::Text(format!("Word {}. ", i)))
                    .await
                    .unwrap();
            }
            in_tx.send(ChunkerInput::Stop).await.unwrap();

            while out_rx.recv().await.is_some() {}
        });
    });

    group.finish();
}

/// Benchmark audio operations
fn bench_audio_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_operations");

    let sample_rates = vec![16000, 22050, 32000, 44100];
    let durations_secs = vec![1, 5, 10];

    for &sample_rate in &sample_rates {
        for &duration in &durations_secs {
            let num_samples = sample_rate * duration;
            let audio: Vec<f32> = (0..num_samples)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                })
                .collect();

            let bench_name = format!("pcm_to_i16/{}Hz_{}s", sample_rate, duration);
            group.throughput(Throughput::Bytes(num_samples as u64 * 4));
            group.bench_function(bench_name, |b| {
                b.iter(|| {
                    let i16_samples: Vec<i16> = audio
                        .iter()
                        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                        .collect();
                    black_box(i16_samples);
                });
            });

            let bench_name = format!("bytes_conversion/{}Hz_{}s", sample_rate, duration);
            group.throughput(Throughput::Bytes(num_samples as u64 * 4));
            group.bench_function(bench_name, |b| {
                b.iter(|| {
                    let bytes: Vec<u8> = audio.iter().flat_map(|&s| s.to_le_bytes()).collect();
                    black_box(bytes);
                });
            });
        }
    }

    group.finish();
}

/// Benchmark protocol serialization
fn bench_protocol_serialization(c: &mut Criterion) {
    use soprano::protocol::{AudioHeader, ClientMessage, ServerMessage};

    let mut group = c.benchmark_group("protocol");

    group.bench_function("serialize_client_config", |b| {
        let msg = ClientMessage::Config {
            engine: Some(soprano::config::EngineId::Soprano),
            voice_path: Some("/path/to/voice".to_string()),
            speed: Some(1.0),
            language_id: Some("en".to_string()),
            min_chars: Some(24),
            min_words: Some(2),
            max_chars: Some(160),
            max_delay_ms: Some(220),
        };
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&msg));
        });
    });

    group.bench_function("serialize_client_text", |b| {
        let msg = ClientMessage::Text {
            text: "Hello world, this is a test message.".to_string(),
        };
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&msg));
        });
    });

    group.bench_function("serialize_server_ready", |b| {
        let msg = ServerMessage::Ready {
            session_id: "test-session-12345".to_string(),
        };
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&msg));
        });
    });

    group.bench_function("serialize_audio_header", |b| {
        let header = AudioHeader::with_text(
            42,
            32000,
            1,
            "pcm_s16le",
            "Hello world, this is synthesized text",
        );
        b.iter(|| {
            let _ = serde_json::to_string(black_box(&header));
        });
    });

    let json_text = r#"{"type":"text","text":"Hello world"}"#;
    group.bench_function("deserialize_client_text", |b| {
        b.iter(|| {
            let _: ClientMessage = serde_json::from_str(black_box(json_text)).unwrap();
        });
    });

    group.finish();
}

/// Comprehensive pipeline benchmark
fn bench_full_pipeline(c: &mut Criterion) {
    let rt = tokio_runtime();

    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("text_to_audio", |b| {
        b.to_async(&rt).iter(|| async {
            let text = "Hello world! Dr. Smith has 5 patients today.";
            let normalized = clean_text(text);
            let chunks = split_and_recombine_text(&normalized, 100, 200);

            let config = Arc::new(tokio::sync::RwLock::new(test_config()));
            let (in_tx, in_rx) = tokio::sync::mpsc::channel(8);
            let (out_tx, mut out_rx) = tokio::sync::mpsc::channel(8);

            tokio::spawn(run_chunker(in_rx, out_tx, config));

            for chunk in chunks {
                in_tx.send(ChunkerInput::Text(chunk + " ")).await.unwrap();
            }
            in_tx.send(ChunkerInput::Stop).await.unwrap();

            while out_rx.recv().await.is_some() {}

            black_box(normalized);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_normalization,
    bench_text_splitter,
    bench_chunker,
    bench_audio_operations,
    bench_protocol_serialization,
    bench_full_pipeline
);

criterion_main!(benches);

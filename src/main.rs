//! Soprano TTS - Unified CLI Tool
//!
//! A single binary providing:
//! - `serve` - Run the TTS server
//! - `download` - Download models from HuggingFace
//! - `list` - List available models
//! - `cache` - Show cache information
//! - `generate` - Generate audio from text
//!
//! Usage:
//!   soprano-tts serve --port 8080
//!   soprano-tts download ekwek/Soprano-1.1-80M
//!   soprano-tts list
//!   soprano-tts cache
//!   soprano-tts generate --text "Hello world"

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use hound::{WavSpec, WavWriter};
use tokio::net::TcpListener;
use tokio::signal;
use tracing::{error, info, warn, Level};

use soprano_tts::{
    config::{
        init_tracing, load_dotenv, parse_device, Cli, Commands, DownloadArgs, GenerateArgs,
        GenerationConfig, ServeArgs, StreamConfig,
    },
    model_loader::{cuda_available, list_available_models, ModelCache},
    normalization::clean_text,
    server::{self, AppState},
    splitter::split_and_recombine_text,
    tts::{self, SopranoTtsEngineBuilder, TtsEngine, TtsRequest},
};

/// Main entry point with subcommand dispatch
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file if present
    load_dotenv();

    // Parse command line arguments
    let cli = Cli::parse();

    // Determine which command to run
    match cli.command {
        Some(Commands::Serve(args)) => run_server(args).await,
        Some(Commands::Download(args)) => run_download(args).await,
        Some(Commands::List) => run_list().await,
        Some(Commands::Cache) => run_cache(None).await,
        Some(Commands::Generate(args)) => run_generate(args).await,
        None => {
            // Backward compatibility: if no subcommand, default to serve
            // using the legacy args from the top-level CLI
            let serve_args = cli.to_serve_args();
            run_server(serve_args).await
        }
    }
}

/// Run the TTS server
async fn run_server(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing/logging
    init_tracing(&args.log);

    info!(
        version = env!("CARGO_PKG_VERSION"),
        "starting soprano-tts server"
    );

    // Log configuration
    info!(
        host = %args.host,
        port = args.port,
        device = %args.device,
        workers = args.workers,
        model_id = %args.model_id,
        "configuration loaded"
    );

    // Parse device configuration
    let device = parse_device(&args.device);
    info!(device = ?device, "using device");

    // Initialize TTS engine
    info!("initializing TTS engine...");
    let tts_engine = match initialize_tts_engine(&args, device).await {
        Ok(engine) => {
            info!("TTS engine initialized successfully");
            engine
        }
        Err(e) => {
            error!(error = %e, "failed to initialize TTS engine");
            return Err(e);
        }
    };

    // Create stream configuration from CLI args
    let mut stream_config = StreamConfig::default();
    stream_config.chunker.min_words = args.min_words.max(1);
    stream_config.generation.temperature = args.temperature;
    stream_config.generation.top_p = args.top_p;
    stream_config.generation.repetition_penalty = args.repetition_penalty;

    // Create application state
    let state = AppState::new(
        tts_engine,
        stream_config,
        args.tts_inflight(),
        args.include_text,
    );

    // Bind TCP listener
    let addr = format!("{}:{}", args.host, args.port);
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => {
            info!(address = %addr, "server bound to address");
            l
        }
        Err(e) => {
            error!(address = %addr, error = %e, "failed to bind to address");
            return Err(e.into());
        }
    };

    // Create shutdown signal handler
    let shutdown = shutdown_signal();

    // Start server with graceful shutdown
    info!("server ready - accepting connections");
    if let Err(e) = server::serve_with_shutdown(listener, state, shutdown).await {
        error!(error = %e, "server error");
        return Err(e);
    }

    info!("server exited cleanly");
    Ok(())
}

/// Initialize the TTS engine based on CLI arguments
async fn initialize_tts_engine(
    args: &ServeArgs,
    device: candle_core::Device,
) -> Result<Arc<dyn TtsEngine>, Box<dyn std::error::Error>> {
    // Check if model download is enabled and needed
    let model_path = if args.download {
        let cache_dir = args.cache_dir();
        let model_dir = cache_dir.join(args.model_id.replace('/', "--"));

        if !model_dir.exists() {
            info!(model_id = %args.model_id, "model not found in cache, downloading...");

            #[cfg(feature = "model-download")]
            {
                download_model_simple(&args.model_id, &cache_dir).await?;
            }

            #[cfg(not(feature = "model-download"))]
            {
                warn!("model-download feature not enabled, using local model only");
                return Err("Model not found and download feature not enabled".into());
            }
        }

        model_dir
    } else {
        args.model_path.clone()
    };

    info!(model_path = %model_path.display(), "using model path");

    // Create TTS engine config
    let engine_config = tts::SopranoEngineConfig {
        model_path: model_path.clone(),
        device: device.clone(),
        num_workers: args.workers,
        sample_rate: args.sample_rate,
        ..Default::default()
    };

    // Create TTS engine
    let engine = tts::SopranoTtsEngine::new(engine_config).await?;

    Ok(Arc::new(engine))
}

/// Simple model download for server initialization
#[cfg(feature = "model-download")]
async fn download_model_simple(
    model_id: &str,
    cache_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use futures_util::StreamExt;
    use indicatif::{ProgressBar, ProgressStyle};
    use tokio::io::AsyncWriteExt;

    info!(model_id = %model_id, cache_dir = %cache_dir.display(), "downloading model");

    let client = reqwest::Client::new();
    let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);

    // Files to download (Soprano needs both the LLM + decoder)
    let files = vec![
        "model.safetensors",
        "tokenizer.json",
        "config.json",
        "decoder.pth",
    ];

    // Create cache directory
    tokio::fs::create_dir_all(cache_dir).await?;

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let path = cache_dir.join(file);

        if path.exists() {
            info!(file = %file, "already exists, skipping");
            continue;
        }

        info!(file = %file, url = %url, "downloading");

        let response = client.get(&url).send().await?;
        let total_size = response.content_length().unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut file_handle = tokio::fs::File::create(&path).await?;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk: bytes::Bytes = chunk?;
            file_handle.write_all(&chunk).await?;
            pb.inc(chunk.len() as u64);
        }

        pb.finish_with_message("done");
        info!(file = %file, path = %path.display(), "download complete");
    }

    Ok(())
}

/// Run the download subcommand
async fn run_download(args: DownloadArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let level = Level::INFO;
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    info!("Soprano TTS - Model Download");

    // Handle cleanup command
    if args.cleanup {
        return clean_cache(args.cache_dir.as_deref(), args.max_age).await;
    }

    // Handle verify command
    if args.verify {
        if let Some(ref model_id) = args.model_id {
            return verify_model(model_id, args.cache_dir.as_deref()).await;
        } else {
            eprintln!("Error: --verify requires a MODEL_ID");
            std::process::exit(1);
        }
    }

    // Default: download model
    if let Some(ref model_id) = args.model_id {
        return download_single_model(model_id, &args).await;
    }

    // No command specified, show help
    println!("Usage: soprano-tts download <MODEL_ID>");
    println!();
    println!("Use --help for more information");
    println!("Use 'soprano-tts list' to see available models");

    Ok(())
}

/// Download a single model
#[cfg(feature = "model-download")]
async fn download_single_model(
    model_id: &str,
    args: &DownloadArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use soprano_tts::model_loader::{download_model, DownloadConfig};

    info!("Downloading model: {}", model_id);
    info!("CUDA available: {}", cuda_available());

    // Determine cache directory
    let cache_path = match &args.cache_dir {
        Some(path) => path.to_path_buf(),
        None => ModelCache::default_cache_dir()?,
    };

    info!("Cache directory: {:?}", cache_path);

    // Parse files if specified
    let files: Option<Vec<&str>> = args.files.as_ref().map(|f| {
        f.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    });

    // Build download config
    let config = DownloadConfig {
        hf_endpoint: args.hf_endpoint.clone(),
        show_progress: !args.no_progress,
        max_retries: args.retries,
        timeout_secs: args.timeout,
        verify_checksums: !args.no_verify,
        ..Default::default()
    };

    // Download the model
    let model_path = download_model(
        model_id,
        &cache_path,
        files.as_deref(),
        Some(config),
    )
    .await
    .map_err(|e| format!("Failed to download model: {}", e))?;

    info!("Model downloaded successfully to: {:?}", model_path);

    // List downloaded files
    println!("\nDownloaded files:");
    for entry in std::fs::read_dir(&model_path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        let size_mb = metadata.len() as f64 / 1_048_576.0;
        println!(
            "  {} ({:.2} MB)",
            entry.file_name().to_string_lossy(),
            size_mb
        );
    }

    // Verify model can be loaded
    info!("Verifying model can be loaded...");

    match soprano_tts::model_loader::load_model_from_path(&model_path) {
        Ok(format) => {
            info!("Model verified successfully! Format: {:?}", format);
        }
        Err(e) => {
            warn!("Model downloaded but could not be verified: {}", e);
            warn!("The model may be incomplete or corrupted.");
        }
    }

    Ok(())
}

#[cfg(not(feature = "model-download"))]
async fn download_single_model(
    _model_id: &str,
    _args: &DownloadArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Error: model-download feature not enabled");
    eprintln!("Compile with: cargo build --features model-download");
    std::process::exit(1);
}

/// Run the list subcommand
async fn run_list() -> Result<(), Box<dyn std::error::Error>> {
    println!("Available Soprano TTS Models:");
    println!();

    let models = list_available_models();

    for model in models {
        println!("  Model ID:  {}", model.model_id);
        println!("  Description: {}", model.description);
        println!("  Size: ~{} MB", model.size_mb);
        println!("  Tags: {}", model.tags.join(", "));
        println!();
    }

    println!("Download a model with:");
    println!("  soprano-tts download <MODEL_ID>");

    Ok(())
}

/// Run the cache subcommand
async fn run_cache(cache_dir: Option<&std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
    let cache_path = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => ModelCache::default_cache_dir()?,
    };

    println!("Cache Directory: {:?}", cache_path);

    if !cache_path.exists() {
        println!("Cache directory does not exist yet.");
        return Ok(());
    }

    let cache = ModelCache::new(&cache_path)?;
    let mut total_size: u64 = 0;
    let mut model_count = 0;

    for entry in std::fs::read_dir(&cache_path)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let model_id = entry.file_name().to_string_lossy().to_string();
            let size = cache.get_cache_size(&model_id)?;
            let size_mb = size as f64 / 1_048_576.0;

            println!("  {}: {:.2} MB", model_id.replace("--", "/"), size_mb);

            total_size += size;
            model_count += 1;
        }
    }

    println!();
    println!("Total models: {}", model_count);
    println!("Total size: {:.2} MB", total_size as f64 / 1_048_576.0);

    Ok(())
}

/// Clean old cached models
async fn clean_cache(
    cache_dir: Option<&std::path::Path>,
    max_age: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let cache_path = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => ModelCache::default_cache_dir()?,
    };

    info!("Cleaning cache at {:?}", cache_path);
    info!("Removing models older than {} days", max_age);

    let cache = ModelCache::new(&cache_path)?;

    // Get list of models to keep (none for now - could be expanded)
    let keep_models: Vec<&str> = vec![];

    let removed = cache
        .cleanup_old_models(&keep_models, max_age)
        .map_err(|e| format!("Failed to clean cache: {}", e))?;

    println!("Removed {} old model(s)", removed);

    Ok(())
}

/// Verify a downloaded model
async fn verify_model(
    model: &str,
    cache_dir: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let cache_path = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => ModelCache::default_cache_dir()?,
    };

    // Check if model is a path or model ID
    let model_path = if std::path::Path::new(model).exists() {
        std::path::PathBuf::from(model)
    } else {
        cache_path.join(model.replace('/', "--"))
    };

    if !model_path.exists() {
        eprintln!("Error: Model not found at: {:?}", model_path);
        std::process::exit(1);
    }

    println!("Verifying model at: {:?}", model_path);

    // Check required files
    let required_files = vec!["tokenizer.json", "config.json"];

    let mut missing = vec![];
    for file in &required_files {
        if !model_path.join(file).exists() {
            missing.push(file);
        }
    }

    if !missing.is_empty() {
        println!("Warning: Missing required files: {:?}", missing);
    } else {
        println!("All required files present");
    }

    // Check for model weights
    let weight_files = vec!["model.safetensors", "model.gguf", "pytorch_model.bin"];
    let mut found_weights = false;

    for file in &weight_files {
        if model_path.join(file).exists() {
            println!("Found weight file: {}", file);
            found_weights = true;
            break;
        }
    }

    if !found_weights {
        println!("Warning: No model weight files found");
    }

    // Try to load the model
    info!("Attempting to verify model format...");

    match soprano_tts::model_loader::load_model_from_path(&model_path) {
        Ok(format) => {
            println!("Model verified successfully!");
            println!("Format: {:?}", format);
        }
        Err(e) => {
            eprintln!("Failed to verify model: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Run the generate subcommand
async fn run_generate(args: GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════╗");
    println!("║     Soprano TTS Sample Generator       ║");
    println!("╚════════════════════════════════════════╝");
    println!();

    // Determine model path (always real model)
    let model_path = args
        .model_path
        .clone()
        .or_else(get_default_model_path)
        .ok_or_else(|| {
            "Model not found. Download with: soprano-tts download ekwek/Soprano-1.1-80M".to_string()
        })?;

    // Determine device
    let requested_device = parse_device(&args.device);
    let device = if args.device.to_lowercase() == "cuda" && !cuda_available() {
        println!("Warning: CUDA not available, falling back to CPU");
        candle_core::Device::Cpu
    } else {
        requested_device
    };

    // Show configuration
    println!("Configuration:");
    println!("  Mode: Real Model");
    println!("  Model Path: {:?}", model_path);
    println!("  Device: {:?}", device);
    println!("  Sample Rate: {} Hz", args.sample_rate);
    println!("  Temperature: {}", args.temperature);
    println!("  Top-p: {}", args.top_p);
    println!("  Repetition Penalty: {}", args.repetition_penalty);
    println!("  Speed: {}", args.speed);
    println!("  Workers: {}", args.workers);
    println!("  Variations: {}", args.variations);

    // Validate model path
    if !model_path.exists() {
        eprintln!("\nError: Model path does not exist: {:?}", model_path);
        eprintln!("\nPlease download the model first:");
        eprintln!("  soprano-tts download ekwek/Soprano-1.1-80M");
        std::process::exit(1);
    }

    // Check for required files
    let required_files = ["model.safetensors", "tokenizer.json", "config.json", "decoder.pth"];
    let mut missing = Vec::new();
    for file in &required_files {
        if !model_path.join(file).exists() {
            missing.push(*file);
        }
    }

    if !missing.is_empty() {
        eprintln!("\nError: Model directory is missing required files:");
        for file in &missing {
            eprintln!("  - {}", file);
        }
        eprintln!("\nPlease download the model first:");
        eprintln!("  soprano-tts download ekwek/Soprano-1.1-80M");
        std::process::exit(1);
    }

    println!();

    // Collect texts to process
    let texts: Vec<String> = if let Some(file_path) = args.file {
        println!("Loading texts from {:?}...", file_path);
        let content = tokio::fs::read_to_string(&file_path).await?;
        content
            .lines()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else if let Some(text) = args.text {
        vec![text]
    } else {
        // Default samples
        vec![
            "Hello, world! This is Soprano TTS.".to_string(),
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "Soprano is a high-performance text-to-speech model.".to_string(),
            "Rust provides memory safety without garbage collection.".to_string(),
        ]
    };

    println!("Processing {} text samples...", texts.len());
    println!();

    // Create output directory if needed
    let output_dir = if texts.len() > 1 || args.variations > 1 {
        if args.output.extension().is_some() {
            // If output has extension, treat as directory
            args.output.with_extension("")
        } else {
            args.output.clone()
        }
    } else {
        args.output
            .parent()
            .unwrap_or(&PathBuf::from("."))
            .to_path_buf()
    };

    if texts.len() > 1 || args.variations > 1 {
        tokio::fs::create_dir_all(&output_dir).await?;
    }

    // Generate samples
    let mut success_count = 0;
    let mut error_count = 0;

    for (idx, text) in texts.iter().enumerate() {
        for variation in 0..args.variations {
            let output_path = if texts.len() == 1 && args.variations == 1 {
                args.output.clone()
            } else if args.variations == 1 {
                output_dir.join(format!("sample_{:03}.wav", idx + 1))
            } else {
                output_dir.join(format!("sample_{:03}_v{}.wav", idx + 1, variation + 1))
            };

            match generate_sample(
                text,
                &output_path,
                args.sample_rate,
                args.temperature,
                args.max_new_tokens,
                args.top_p,
                args.repetition_penalty,
                args.speed,
                args.verbose,
                variation,
                &model_path,
                &device,
                args.workers,
            )
            .await
            {
                Ok(()) => success_count += 1,
                Err(e) => {
                    eprintln!("✗ Error generating sample: {}", e);
                    error_count += 1;
                }
            }
        }
    }

    println!();
    println!("╔════════════════════════════════════════╗");
    println!("║           Generation Complete          ║");
    println!("╚════════════════════════════════════════╝");
    println!();
    println!("Results:");
    println!("  ✓ Successful: {}", success_count);
    if error_count > 0 {
        println!("  ✗ Failed: {}", error_count);
    }
    println!();

    if success_count > 0 {
        println!("Output location: {:?}", output_dir);
        println!();
        println!("You can play the generated files with:");
        println!("  - aplay sample.wav    (Linux ALSA)");
        println!("  - afplay sample.wav   (macOS)");
        println!("  - ffplay sample.wav   (cross-platform)");
        println!();

        println!("Note: Audio generated using the real Soprano TTS model.");
    }

    Ok(())
}

/// Get the default model path from cache
fn get_default_model_path() -> Option<PathBuf> {
    let cache = ModelCache::default_cache_dir().ok()?;
    let model_id = "ekwek/Soprano-1.1-80M";
    let model_path = cache.join(model_id.replace('/', "--"));

    if model_path.exists() {
        Some(model_path)
    } else {
        None
    }
}

/// Generate a single audio sample
#[allow(clippy::too_many_arguments)]
async fn generate_sample(
    text: &str,
    output_path: &PathBuf,
    sample_rate: u32,
    temperature: f32,
    max_new_tokens: usize,
    top_p: f32,
    repetition_penalty: f32,
    speed: f32,
    verbose: bool,
    variation: usize,
    model_path: &PathBuf,
    device: &candle_core::Device,
    workers: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();

    if verbose {
        println!("Processing text: {}", text);
        println!("Preprocessing...");
    }

    // Preprocess text
    let processed_text = preprocess_text(text);

    if verbose {
        println!("Cleaned text: {}", processed_text);
    }

    let _ = variation;

    // Generate audio (always real model)
    let audio = generate_with_model(
        &processed_text,
        model_path,
        device.clone(),
        workers,
        temperature,
        max_new_tokens,
        top_p,
        repetition_penalty,
        speed,
        verbose,
    )
    .await?;

    if verbose {
        println!("Generated {} samples", audio.len());

        // Calculate RMS amplitude for debugging
        let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let peak = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        println!("Audio stats: RMS={:.4}, Peak={:.4}", rms, peak);
        println!("Saving to {:?}...", output_path);
    }

    // Save to file
    save_wav(&audio, sample_rate, output_path)?;

    let elapsed = start.elapsed();
    let duration_secs = audio.len() as f32 / sample_rate as f32;

    println!(
        "✓ Generated: {:?} ({} samples, {:.2}s audio, {:.2}s elapsed)",
        output_path,
        audio.len(),
        duration_secs,
        elapsed.as_secs_f32()
    );

    Ok(())
}

/// Process text through normalization pipeline
fn preprocess_text(text: &str) -> String {
    // Clean the text using the normalization pipeline
    let cleaned = clean_text(text);

    // Split into sentences if needed
    let sentences = split_and_recombine_text(&cleaned, 50, 300);

    // Join with proper spacing
    sentences.join(" ")
}

/// Generate audio using the real TTS model
#[allow(clippy::too_many_arguments)]
async fn generate_with_model(
    text: &str,
    model_path: &PathBuf,
    device: candle_core::Device,
    workers: usize,
    temperature: f32,
    max_new_tokens: usize,
    top_p: f32,
    repetition_penalty: f32,
    speed: f32,
    verbose: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if verbose {
        println!("Loading model from {:?}...", model_path);
        println!("Device: {:?}", device);
        println!("Workers: {}", workers);
    }

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
        min_new_tokens: 32,
    };

    // Build TTS engine
    let init_start = Instant::now();

    let engine = SopranoTtsEngineBuilder::new(model_path)
        .with_device(device)
        .with_workers(workers)
        .with_generation_config(gen_config)
        .build()
        .await
        .map_err(|e| format!("Failed to initialize TTS engine: {}", e))?;

    if verbose {
        println!("Model loaded in {:?}", init_start.elapsed());

        // Get engine info
        let info = engine.info();
        println!("Engine: {} v{}", info.name, info.version);
        println!("Sample rate: {} Hz", info.sample_rate);
    }

    // Create TTS request
    let request = TtsRequest::new(text).with_speed(speed).with_language("en");

    // Synthesize
    if verbose {
        println!("Synthesizing...");
    }

    let synthesis_start = Instant::now();
    let response = engine
        .synthesize(request)
        .await
        .map_err(|e| format!("Synthesis failed: {}", e))?;

    let synthesis_duration = synthesis_start.elapsed();

    if verbose {
        println!("Synthesis completed in {:?}", synthesis_duration);
        println!(
            "Generated {} samples ({:.2}s audio)",
            response.num_samples, response.duration_secs
        );
        println!("Tokens generated: {}", response.metadata.tokens_generated);
        println!("Processing time: {}ms", response.metadata.processing_time_ms);
        println!("LLM time: {}ms", response.metadata.llm_time_ms);
        println!("Decoder time: {}ms", response.metadata.decoder_time_ms);

        // Calculate RTF
        if synthesis_duration.as_secs_f32() > 0.0 && response.duration_secs > 0.0 {
            let rtf = synthesis_duration.as_secs_f32() / response.duration_secs;
            println!("Real-time factor: {:.3}", rtf);
        }
    }

    Ok(response.pcm)
}

/// Save audio samples to WAV file with proper formatting
fn save_wav(
    samples: &[f32],
    sample_rate: u32,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;

    // Apply dithering to reduce quantization noise
    let dither_amount = 1.0 / (1u32 << 16) as f32; // Very small amount

    for (i, sample) in samples.iter().enumerate() {
        // Add triangular dither
        let dither = if i % 2 == 0 {
            dither_amount
        } else {
            -dither_amount
        };

        let dithered = sample + dither;

        // Convert f32 [-1.0, 1.0] to i16 with proper clamping
        let amplitude = i16::MAX as f32;
        let sample_i16 = (dithered.clamp(-1.0, 1.0) * amplitude) as i16;
        writer.write_sample(sample_i16)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Create a future that resolves on shutdown signal (SIGTERM or SIGINT)
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("received Ctrl+C, shutting down");
        }
        _ = terminate => {
            info!("received SIGTERM, shutting down");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::parse_from(["soprano-tts", "serve", "--port", "8888"]);
        assert!(matches!(cli.command, Some(Commands::Serve(args)) if args.port == 8888));
    }

    #[test]
    fn test_cli_device_options() {
        let cli_cuda = Cli::parse_from(["soprano-tts", "serve", "--device", "cuda"]);
        assert!(
            matches!(cli_cuda.command, Some(Commands::Serve(args)) if args.device == "cuda")
        );

        let cli_cpu = Cli::parse_from(["soprano-tts", "serve", "--device", "cpu"]);
        assert!(
            matches!(cli_cpu.command, Some(Commands::Serve(args)) if args.device == "cpu")
        );
    }

    #[test]
    fn test_save_wav() {
        let samples = vec![0.0f32; 1000];
        let temp_path = PathBuf::from("/tmp/test_soprano_sample.wav");

        save_wav(&samples, 16000, &temp_path).unwrap();

        // Verify file exists and has correct size
        let metadata = std::fs::metadata(&temp_path).unwrap();
        assert!(metadata.len() > 0);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}

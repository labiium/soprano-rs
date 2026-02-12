//! Debug Comparison Example
//!
//! This example demonstrates how to use the debug mode to save intermediate
//! tensors for layer-by-layer comparison with the Python implementation.
//!
//! Usage:
//!   cargo run --example debug_comparison -- --model-dir /path/to/model
//!
//! The example will:
//! 1. Load the model with debug mode enabled
//! 2. Run inference on a test prompt
//! 3. Save all intermediate tensors to /tmp/layer_comparison/rust/
//!
//! You can then compare these with Python tensors using:
//!   python layer_by_layer_comparison.py --rust-dir /tmp/layer_comparison/rust

use std::path::PathBuf;

use candle_core::Device;
use clap::Parser;
use soprano::model::{DebugConfig, GenerationConfig, SopranoModel};

#[derive(Parser)]
#[command(name = "debug_comparison")]
#[command(about = "Generate intermediate tensors for layer-by-layer comparison")]
struct Args {
    /// Path to model directory containing model.safetensors, config.json, tokenizer.json
    #[arg(long, default_value = "model")]
    model_dir: PathBuf,

    /// Output directory for debug tensors
    #[arg(long, default_value = "/tmp/layer_comparison/rust")]
    output_dir: PathBuf,

    /// Test prompt
    #[arg(short, long, default_value = "[STOP][TEXT]Hello world[START]")]
    prompt: String,

    /// Max new tokens to generate
    #[arg(long, default_value_t = 1)]
    max_new_tokens: usize,

    /// Temperature for generation (0.0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Top-p sampling parameter
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         DEBUG COMPARISON - Layer-by-Layer Tensor Capture         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model directory: {}", args.model_dir.display());
    println!("Output directory: {}", args.output_dir.display());
    println!("Prompt: {}", args.prompt);
    println!();

    // Setup device (CPU for consistent results with Python)
    let device = Device::Cpu;
    println!("Using device: {:?}", device);

    // Create debug configuration
    let debug_config = DebugConfig::new(&args.output_dir).enable_all();
    println!("Debug mode: ENABLED");
    println!("  - Save embeddings: {}", debug_config.save_embeddings);
    println!("  - Save layer I/O: {}", debug_config.save_layer_io);
    println!("  - Save attention: {}", debug_config.save_attention);
    println!("  - Save MLP: {}", debug_config.save_mlp);
    println!("  - Save final outputs: {}", debug_config.save_final);
    println!();

    // Load model with debug config
    println!("Loading model...");
    let mut model = SopranoModel::from_path_with_debug(&args.model_dir, device, debug_config)?;
    println!("✓ Model loaded successfully");
    println!("  EOS token ID: {}", model.eos_token_id());
    println!();

    // Setup generation config
    let gen_config = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        min_new_tokens: 0, // Allow immediate stop for testing
    };

    println!("Generation config:");
    println!("  Max new tokens: {}", gen_config.max_new_tokens);
    println!("  Temperature: {}", gen_config.temperature);
    println!("  Top-p: {}", gen_config.top_p);
    println!("  Repetition penalty: {}", gen_config.repetition_penalty);
    println!();

    // Run generation
    println!("Running inference...");
    println!("─────────────────────────────────────────────────────────────────");
    let result = model.generate(&args.prompt, &gen_config)?;
    println!("─────────────────────────────────────────────────────────────────");
    println!();

    // Print results
    println!("Generation complete!");
    println!("  Tokens generated: {}", result.token_count);
    println!("  Finish reason: {:?}", result.finish_reason);
    println!("  Hidden states shape: {:?}", result.hidden_states.shape());
    println!();

    // List saved tensors
    println!("Saved tensors:");
    if let Ok(entries) = std::fs::read_dir(&args.output_dir) {
        let mut files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "npy")
                    .unwrap_or(false)
            })
            .collect();
        files.sort_by_key(|a| a.file_name());

        for entry in files {
            let path = entry.path();
            let name = path.file_stem().unwrap_or_default().to_string_lossy();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            println!("  ✓ {} ({:.2} KB)", name, size as f64 / 1024.0);
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                      NEXT STEPS                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("1. Run Python extraction:");
    println!("   python layer_by_layer_comparison.py --output-dir /tmp/layer_comparison");
    println!();
    println!("2. Compare tensors:");
    println!("   python layer_by_layer_comparison.py \\");
    println!("     --output-dir /tmp/layer_comparison \\");
    println!("     --rust-dir /tmp/layer_comparison/rust");
    println!();
    println!("3. View detailed comparison report:");
    println!("   cat /tmp/layer_comparison/comparison_report.json");
    println!();

    Ok(())
}

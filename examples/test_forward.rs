//! Test full model forward pass

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct Qwen3ConfigFile {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    max_window_layers: usize,
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    use_sliding_window: bool,
    hidden_act: String,
    eos_token_id: usize,
}

fn load_config(model_dir: &Path) -> (qwen2::Config, usize) {
    let cfg_path = model_dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path).expect("Failed to read config.json");
    let raw: Qwen3ConfigFile = serde_json::from_str(&cfg_str).expect("Failed to parse config.json");

    let sliding_window = raw.sliding_window.unwrap_or(raw.max_position_embeddings);

    let cfg = qwen2::Config {
        vocab_size: raw.vocab_size,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        max_position_embeddings: raw.max_position_embeddings,
        sliding_window,
        max_window_layers: raw.max_window_layers,
        tie_word_embeddings: raw.tie_word_embeddings,
        rope_theta: raw.rope_theta,
        rms_norm_eps: raw.rms_norm_eps,
        use_sliding_window: raw.use_sliding_window,
        hidden_act: match raw.hidden_act.as_str() {
            "silu" | "swish" => candle_nn::Activation::Silu,
            "gelu" => candle_nn::Activation::Gelu,
            _ => candle_nn::Activation::Silu,
        },
    };

    (cfg, raw.eos_token_id)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_dir =
        std::path::PathBuf::from("/home/emmanuel/.cache/soprano-rs/models/ekwek--Soprano-1.1-80M");

    println!("Loading config...");
    let (config, _eos_token_id) = load_config(&model_dir);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_attention_heads: {}", config.num_attention_heads);
    println!("  num_key_value_heads: {}", config.num_key_value_heads);

    println!("Loading model...");
    let st_path = model_dir.join("model.safetensors");
    let bias_path = model_dir.join("model.attn_biases.safetensors");

    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[st_path, bias_path], DType::F32, &device)? };

    // Create model
    let mut model = qwen2::Model::new(&config, vb.clone())?;
    let lm_head = candle_nn::linear_no_bias(512, 8192, vb.pp("lm_head"))?;

    // Test input: first few tokens of "Hello world"
    let input_ids = Tensor::new(&[3u32, 1, 8077], &device)?.unsqueeze(0)?;
    println!("Input shape: {:?}", input_ids.shape());

    // Forward pass
    let hidden = model.forward(&input_ids, 0, None)?;
    println!("Hidden output shape: {:?}", hidden.shape());

    // Get last token hidden state
    let last_hidden = hidden.narrow(1, 2, 1)?; // Last of 3 tokens
    let last_hidden = last_hidden.squeeze(1)?;
    println!("Last hidden shape: {:?}", last_hidden.shape());

    // Apply LM head
    let logits = last_hidden.apply(&lm_head)?;
    println!("Logits shape: {:?}", logits.shape());

    // Get top 5 tokens
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_1d = logits_f32.squeeze(0)?;
    let logits_vec: Vec<f32> = logits_1d.to_vec1()?;

    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 predicted tokens:");
    for (i, (token_id, score)) in indexed.iter().take(10).enumerate() {
        println!("  {}. Token {}: logit={:.4}", i + 1, token_id, score);
    }

    // Print statistics
    let min_val = logits_1d.min_all()?.to_scalar::<f32>()?;
    let max_val = logits_1d.max_all()?.to_scalar::<f32>()?;
    let mean_val = logits_1d.mean_all()?.to_scalar::<f32>()?;
    println!(
        "\nLogits stats: min={:.4}, max={:.4}, mean={:.4}",
        min_val, max_val, mean_val
    );

    // Print hidden state stats
    let hidden_f32 = hidden.to_dtype(DType::F32)?;
    let h_min = hidden_f32.min_all()?.to_scalar::<f32>()?;
    let h_max = hidden_f32.max_all()?.to_scalar::<f32>()?;
    let h_mean = hidden_f32.mean_all()?.to_scalar::<f32>()?;
    println!(
        "Hidden stats: min={:.4}, max={:.4}, mean={:.4}",
        h_min, h_max, h_mean
    );

    // Print first token's hidden state
    let first_hidden = hidden.narrow(1, 0, 1)?.squeeze(1)?;
    let first_vec: Vec<f32> = first_hidden.to_dtype(DType::F32)?.to_vec1()?;
    println!(
        "\nFirst token hidden state (first 10): {:?}",
        &first_vec[..10]
    );

    Ok(())
}

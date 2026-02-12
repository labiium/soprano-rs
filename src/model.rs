//! Soprano language model (LLM) inference.
//!
//! The official Soprano-1.1-80M model on HuggingFace is a `Qwen3ForCausalLM`
//! checkpoint stored as `model.safetensors`.
//!
//! Candle does not provide a dedicated Qwen3 implementation, but the Qwen3 80M
//! checkpoint is compatible with Candle's `qwen2` implementation for the subset
//! of features Soprano uses (standard causal decoding with RoPE + RMSNorm).
//!
//! This module loads the checkpoint with Candle and generates *token hidden
//! states* (last layer) for each generated token, matching the Python reference
//! implementation.

use crate::qwen3;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use safetensors::tensor::{serialize_to_file, Dtype as StDtype, View};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

/// Configuration for token generation.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Minimum number of new tokens before EOS is allowed to stop generation.
    pub min_new_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.0,
            top_p: 0.95,
            repetition_penalty: 1.2,
            min_new_tokens: 32,
        }
    }
}

/// Reason generation finished.
#[derive(Clone, Debug, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Result from generation containing hidden states and metadata.
#[derive(Clone, Debug)]
pub struct GenerationResult {
    /// Hidden states for generated tokens, shape: (T, hidden_size)
    pub hidden_states: Tensor,
    pub finish_reason: FinishReason,
    pub token_count: usize,
}

/// Debug configuration for saving intermediate tensors.
#[derive(Clone, Debug)]
pub struct DebugConfig {
    /// Enable debug mode to save intermediate tensors
    pub enabled: bool,
    /// Directory to save debug tensors
    pub output_dir: std::path::PathBuf,
    /// Save embeddings
    pub save_embeddings: bool,
    /// Save each layer's input/output
    pub save_layer_io: bool,
    /// Save attention outputs
    pub save_attention: bool,
    /// Save MLP outputs
    pub save_mlp: bool,
    /// Save final outputs
    pub save_final: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_dir: std::path::PathBuf::from("/tmp/layer_comparison/rust"),
            save_embeddings: true,
            save_layer_io: true,
            save_attention: true,
            save_mlp: true,
            save_final: true,
        }
    }
}

impl DebugConfig {
    /// Create a new debug config with the specified output directory.
    pub fn new(output_dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            enabled: true,
            output_dir: output_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Enable all debug outputs.
    pub fn enable_all(mut self) -> Self {
        self.enabled = true;
        self.save_embeddings = true;
        self.save_layer_io = true;
        self.save_attention = true;
        self.save_mlp = true;
        self.save_final = true;
        self
    }

    /// Save a tensor to a .npy file.
    pub fn save_tensor(&self, name: &str, tensor: &Tensor) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Ensure output directory exists
        std::fs::create_dir_all(&self.output_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create debug dir: {e}")))?;

        let filepath = self.output_dir.join(format!("rust_{}.npy", name));

        // Convert to F32 for saving
        let tensor_f32 = tensor.to_dtype(DType::F32)?;

        // Save using numpy format
        // First, we need to write the .npy format header and data
        let shape = tensor_f32.shape().dims().to_vec();
        // Flatten tensor to 1D for saving
        let flattened = tensor_f32.flatten_all()?;
        let data = flattened.to_vec1::<f32>()?;

        // Write numpy file
        write_npy_file(&filepath, &shape, &data)?;

        Ok(())
    }
}

/// Write a tensor as a numpy .npy file.
fn write_npy_file(path: &std::path::Path, shape: &[usize], data: &[f32]) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create file: {e}")))?;

    // NPY magic string and version
    let magic: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
    file.write_all(&magic)
        .map_err(|e| candle_core::Error::Msg(format!("Write error: {e}")))?;

    // Version (1.0)
    file.write_all(&[0x01, 0x00])
        .map_err(|e| candle_core::Error::Msg(format!("Write error: {e}")))?;

    // Build header dictionary
    let fortran_order = "False";
    let descr = "<f4"; // Little-endian float32

    // Format shape string
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({},)",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
        descr, fortran_order, shape_str
    );

    // Header must be aligned to 64 bytes
    let header_len = header_dict.len() + 1; // +1 for newline
    let padding_needed = (64 - (10 + header_len) % 64) % 64;
    let padding = " ".repeat(padding_needed);
    let header = format!("{}{}\n", header_dict, padding);

    // Write header length (2 bytes, little-endian)
    let header_len = header.len() as u16;
    file.write_all(&header_len.to_le_bytes())
        .map_err(|e| candle_core::Error::Msg(format!("Write error: {e}")))?;

    // Write header
    file.write_all(header.as_bytes())
        .map_err(|e| candle_core::Error::Msg(format!("Write error: {e}")))?;

    // Write data as little-endian f32
    for &val in data {
        file.write_all(&val.to_le_bytes())
            .map_err(|e| candle_core::Error::Msg(format!("Write error: {e}")))?;
    }

    Ok(())
}

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

fn parse_activation(name: &str) -> candle_nn::Activation {
    match name {
        "silu" | "swish" => candle_nn::Activation::Silu,
        "gelu" => candle_nn::Activation::Gelu,
        "relu" => candle_nn::Activation::Relu,
        _ => candle_nn::Activation::Silu,
    }
}

fn load_qwen3_config(model_dir: &Path) -> Result<(qwen3::Config, usize)> {
    let cfg_path = model_dir.join("config.json");
    let cfg_str = std::fs::read_to_string(&cfg_path)?;
    let raw: Qwen3ConfigFile = serde_json::from_str(&cfg_str)
        .map_err(|e| candle_core::Error::Msg(format!("failed to parse config.json: {e}")))?;

    // Disable sliding window by making it large enough for the model context.
    let sliding_window = raw.sliding_window.unwrap_or(raw.max_position_embeddings);

    let cfg = qwen3::Config {
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
        hidden_act: parse_activation(&raw.hidden_act),
    };

    Ok((cfg, raw.eos_token_id))
}

#[derive(Debug, Clone)]
struct OwnedTensor {
    dtype: StDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> StDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn ensure_qwen2_attn_biases(
    model_dir: &Path,
    cfg: &qwen3::Config,
    device: &Device,
) -> Result<Vec<std::path::PathBuf>> {
    let st_path = model_dir.join("model.safetensors");
    let dtype = match device {
        Device::Cuda(_) => DType::BF16,
        Device::Metal(_) => DType::F16,
        _ => DType::F32,
    };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&st_path), dtype, device)?
    };
    if vb.contains_tensor("model.layers.0.self_attn.q_proj.bias") {
        return Ok(vec![st_path]);
    }

    let out_path = model_dir.join("model.attn_biases.safetensors");
    if !out_path.exists() {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let q_out = cfg.num_attention_heads * head_dim;
        let kv_out = cfg.num_key_value_heads * head_dim;
        let zeros_q = vec![0f32; q_out];
        let zeros_kv = vec![0f32; kv_out];

        let mut tensors: Vec<(String, OwnedTensor)> = Vec::with_capacity(cfg.num_hidden_layers * 3);
        for layer in 0..cfg.num_hidden_layers {
            tensors.push((
                format!("model.layers.{layer}.self_attn.q_proj.bias"),
                OwnedTensor {
                    dtype: StDtype::F32,
                    shape: vec![q_out],
                    data: bytemuck::cast_slice(&zeros_q).to_vec(),
                },
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.k_proj.bias"),
                OwnedTensor {
                    dtype: StDtype::F32,
                    shape: vec![kv_out],
                    data: bytemuck::cast_slice(&zeros_kv).to_vec(),
                },
            ));
            tensors.push((
                format!("model.layers.{layer}.self_attn.v_proj.bias"),
                OwnedTensor {
                    dtype: StDtype::F32,
                    shape: vec![kv_out],
                    data: bytemuck::cast_slice(&zeros_kv).to_vec(),
                },
            ));
        }

        serialize_to_file(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            &None,
            &out_path,
        )
        .map_err(|e| candle_core::Error::Msg(format!("failed to write bias safetensors: {e}")))?;
    }

    Ok(vec![st_path, out_path])
}

/// A Candle-backed Soprano model (Qwen3 compatible with qwen2 implementation).
pub struct SopranoModel {
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    base: qwen3::Model,
    lm_head: candle_nn::Linear,
    debug_config: DebugConfig,
    embeddings: candle_nn::Embedding, // Direct access to embeddings for debugging
}

impl SopranoModel {
    /// Load model from a local directory containing:
    /// - `model.safetensors`
    /// - `config.json`
    /// - `tokenizer.json`
    pub fn from_path(model_dir: &Path, device: Device) -> Result<Self> {
        Self::from_path_with_debug(model_dir, device, DebugConfig::default())
    }

    /// Load model with debug configuration.
    pub fn from_path_with_debug(
        model_dir: &Path,
        device: Device,
        debug_config: DebugConfig,
    ) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(format!("failed to load tokenizer: {e}")))?;

        let (cfg, eos_token_id) = load_qwen3_config(model_dir)?;

        // Load model weights.
        let st_path = model_dir.join("model.safetensors");
        if !st_path.exists() {
            return Err(candle_core::Error::Msg(format!(
                "missing model.safetensors at {}",
                st_path.display()
            )));
        }

        // Use F32 for consistency with Python reference implementation
        // BF16/F16 can cause numerical differences that affect generation quality
        let dtype = DType::F32;

        let paths = ensure_qwen2_attn_biases(model_dir, &cfg, &device)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };

        let base = qwen3::Model::new(&cfg, vb.clone())?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model").pp("embed_tokens"),
        )?;

        Ok(Self {
            tokenizer,
            device,
            eos_token_id: eos_token_id as u32,
            base,
            lm_head,
            debug_config,
            embeddings,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Set debug configuration.
    pub fn set_debug_config(&mut self, config: DebugConfig) {
        self.debug_config = config;
    }

    /// Enable debug mode with the specified output directory.
    pub fn enable_debug(&mut self, output_dir: impl AsRef<std::path::Path>) {
        self.debug_config = DebugConfig::new(output_dir);
    }

    /// Generate hidden states for generated tokens (last layer).
    ///
    /// Returns a tensor of shape `(T, hidden_size)`.
    pub fn generate(&mut self, prompt: &str, cfg: &GenerationConfig) -> Result<GenerationResult> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| candle_core::Error::Msg(format!("tokenization error: {e}")))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(candle_core::Error::Msg("empty prompt".to_string()));
        }

        eprintln!("[GENERATE] Prompt tokens: {:?}", tokens);
        eprintln!("[GENERATE] EOS token ID: {}", self.eos_token_id);

        // Clear KV cache for a fresh generation.
        self.base.clear_kv_cache();

        let temperature = if cfg.temperature <= 0.0 {
            0.001
        } else {
            cfg.temperature
        };
        let sampling = Sampling::TopP {
            p: cfg.top_p as f64,
            temperature: temperature as f64,
        };
        let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

        // Prime cache with the prompt.
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        eprintln!("[GENERATE] Input tensor shape: {:?}", input.shape());
        eprintln!("[GENERATE] Input tokens: {:?}", tokens);

        // Save input_ids in debug mode
        if self.debug_config.enabled {
            self.debug_config.save_tensor("input_ids", &input)?;
        }

        // Get actual embeddings (before any transformer layers)
        let input_embeds = self.embeddings.forward(&input)?;
        if self.debug_config.enabled && self.debug_config.save_embeddings {
            self.debug_config
                .save_tensor("embeddings_output", &input_embeds)?;
        }

        let hidden = self.base.forward(&input, 0, None)?; // (1, seq, hidden)
        let seq_len = hidden.dim(1)?;
        eprintln!("[GENERATE] Hidden shape after prompt: {:?}", hidden.shape());

        let mut last_hidden = hidden.narrow(1, seq_len - 1, 1)?; // (1,1,hidden)
        last_hidden = last_hidden.squeeze(1)?; // (1, hidden)
        let mut logits = last_hidden.apply(&self.lm_head)?; // (1, vocab)

        // Debug: Check logits statistics and top tokens
        let logits_f32 = logits.to_dtype(DType::F32)?;
        let logits_1d = logits_f32.squeeze(0)?;
        let min_logit = logits_1d.min_all()?.to_scalar::<f32>()?;
        let max_logit = logits_1d.max_all()?.to_scalar::<f32>()?;
        eprintln!(
            "[GENERATE] Logits range: [{:.4}, {:.4}]",
            min_logit, max_logit
        );

        // Find top 5 tokens manually
        let logits_vec: Vec<f32> = logits_1d.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("[GENERATE] Top 5 tokens:");
        for (i, (token_id, score)) in indexed.iter().take(5).enumerate() {
            eprintln!("  {}. Token {}: logit={:.4}", i + 1, token_id, score);
        }

        let mut generated_hidden: Vec<Tensor> = Vec::new();
        let mut finish = FinishReason::Length;

        for step in 0..cfg.max_new_tokens {
            // logits is (1, vocab) -> (vocab)
            // Sampling and repetition penalty run on f32 logits.
            let mut logits_1d = logits.squeeze(0)?.to_dtype(DType::F32)?;
            if (cfg.repetition_penalty - 1.0).abs() > f32::EPSILON {
                logits_1d = apply_repetition_penalty(&logits_1d, &tokens, cfg.repetition_penalty)?;
            }
            let next = logits_processor.sample(&logits_1d)?;
            tokens.push(next);
            let maybe_stop =
                next == self.eos_token_id && generated_hidden.len() >= cfg.min_new_tokens;

            // Debug: print every 10 steps and when EOS is detected
            if step % 10 == 0 || next == self.eos_token_id {
                eprintln!(
                    "[GENERATE] step={}, next_token={}, eos={}, hidden_len={}, maybe_stop={}",
                    step,
                    next,
                    self.eos_token_id,
                    generated_hidden.len(),
                    maybe_stop
                );
            }

            // Forward only the new token using KV-cache.
            let tok = Tensor::new(&[next], &self.device)?.unsqueeze(0)?; // (1,1)
            let hidden = self.base.forward(&tok, seq_len + step, None)?; // (1,1,hidden)

            // Save hidden states in debug mode
            if self.debug_config.enabled && self.debug_config.save_layer_io {
                let step_hidden = hidden.squeeze(0)?.squeeze(0)?; // (hidden,)
                self.debug_config.save_tensor(
                    &format!("generated_step_{}_hidden", step),
                    &step_hidden.unsqueeze(0)?,
                )?;
            }

            // Save hidden state for this generated token (skip EOS, matches Python).
            if next != self.eos_token_id {
                let h = hidden.squeeze(0)?.squeeze(0)?; // (hidden)
                generated_hidden.push(h);
            }

            // Compute logits for the next step from this token hidden.
            logits = hidden
                .squeeze(1)?
                .apply(&self.lm_head)?
                .to_dtype(DType::F32)?; // (1,vocab)

            // Save logits in debug mode
            if self.debug_config.enabled && step == 0 && self.debug_config.save_final {
                let logits_squeezed = logits.squeeze(0)?; // (vocab,)
                self.debug_config
                    .save_tensor("logits_output", &logits_squeezed.unsqueeze(0)?)?;
            }

            if maybe_stop {
                finish = FinishReason::Stop;
                break;
            }
        }

        let out = if generated_hidden.is_empty() {
            Tensor::zeros((0, 0), DType::F32, &self.device)?
        } else {
            Tensor::stack(&generated_hidden, 0)?
        };

        eprintln!(
            "[GENERATE] Generated {} tokens. Finish reason: {:?}",
            generated_hidden.len(),
            finish
        );
        eprintln!("[GENERATE] All generated tokens: {:?}", &tokens[11..]); // Skip prompt tokens

        // Save final hidden states in debug mode
        if self.debug_config.enabled && self.debug_config.save_final && !generated_hidden.is_empty()
        {
            self.debug_config.save_tensor("hidden_states_final", &out)?;
        }

        Ok(GenerationResult {
            hidden_states: out.to_dtype(DType::F32)?,
            finish_reason: finish,
            token_count: generated_hidden.len(),
        })
    }

    /// Run a forward pass through the model and capture intermediate outputs.
    ///
    /// This is a debug function that captures outputs at each layer.
    pub fn forward_debug(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        if !self.debug_config.enabled {
            // Just do a regular forward pass
            return self.base.forward(input_ids, 0, None);
        }

        eprintln!("[DEBUG] Running forward pass with debug hooks");
        eprintln!("[DEBUG] Input shape: {:?}", input_ids.shape());

        // Save input
        self.debug_config.save_tensor("input_ids", input_ids)?;

        // Note: Candle's qwen3::Model doesn't expose individual layer outputs
        // directly, so we can only capture the final output here.
        // For full layer-by-layer comparison, we'd need to modify Candle itself
        // or implement the model layers manually.

        let output = self.base.forward(input_ids, 0, None)?;

        eprintln!("[DEBUG] Output shape: {:?}", output.shape());

        // Save output
        self.debug_config
            .save_tensor("embeddings_output", &output)?;

        Ok(output)
    }
}

fn apply_repetition_penalty(logits: &Tensor, tokens: &[u32], penalty: f32) -> Result<Tensor> {
    let mut v: Vec<f32> = logits.to_vec1()?;
    for &t in tokens {
        let idx = t as usize;
        if idx < v.len() {
            if v[idx] > 0.0 {
                v[idx] /= penalty;
            } else {
                v[idx] *= penalty;
            }
        }
    }
    Tensor::new(v, logits.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_config_parses() {
        // This only tests serde conversion logic.
        let json = r#"{
            "vocab_size": 8192,
            "hidden_size": 512,
            "intermediate_size": 2304,
            "num_hidden_layers": 17,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "max_position_embeddings": 1024,
            "sliding_window": null,
            "max_window_layers": 17,
            "tie_word_embeddings": false,
            "rope_theta": 10000,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": false,
            "hidden_act": "silu",
            "eos_token_id": 3
        }"#;
        let raw: Qwen3ConfigFile = serde_json::from_str(json).unwrap();
        assert_eq!(raw.hidden_size, 512);
        assert!(raw.sliding_window.is_none());
    }

    #[test]
    fn debug_config_works() {
        let config = DebugConfig::default();
        assert!(!config.enabled);

        let config = DebugConfig::new("/tmp/test");
        assert!(config.enabled);
        assert_eq!(config.output_dir, std::path::PathBuf::from("/tmp/test"));
    }
}

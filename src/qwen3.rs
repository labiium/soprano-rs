//! Qwen3 (compat) model implementation.
//!
//! The official Soprano checkpoint declares `model_type: qwen3` and does not
//! use attention projection biases (`attention_bias: false`). Candle's built-in
//! `qwen2` implementation uses biases for q/k/v projections.
//!
//! This module is a small compatibility fork of Candle's `qwen2` implementation
//! with **no-bias q/k/v projections**, matching the weights shipped by Soprano.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

use candle_transformers::models::with_tracing::{linear_no_bias, Linear, RmsNorm};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        // Use rope_slow (contiguous-half style) to match HuggingFace Qwen2 implementation.
        // The rope function uses interleaved rotation which produces different results.
        let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        // NOTE: no-bias projections to match Soprano weights.
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        // Qwen3 specific: RMSNorm on Q and K projections
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Qwen3 specific: Apply RMSNorm to Q and K per head
        // Reshape to (b_sz * num_heads, q_len, head_dim) for norm, then reshape back
        let query_states_normed = query_states
            .reshape((b_sz * self.num_heads, q_len, self.head_dim))?
            .apply(&self.q_norm)?
            .reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
        let key_states_normed = key_states
            .reshape((b_sz * self.num_kv_heads, q_len, self.head_dim))?
            .apply(&self.k_norm)?
            .reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;

        let (query_states, key_states) = self.rotary_emb.apply_rotary_emb_qkv(
            &query_states_normed,
            &key_states_normed,
            seqlen_offset,
        )?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states =
            candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        // Try Flash Attention first, fall back to standard attention
        let attn_output = match self.apply_flash_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
        ) {
            Ok(output) => {
                // Flash Attention succeeded
                output
            }
            Err(_) => {
                // Fallback to standard attention
                let scale = 1f64 / f64::sqrt(self.head_dim as f64);
                let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
                let attn_weights = match attention_mask {
                    None => attn_weights,
                    Some(mask) => attn_weights.broadcast_add(mask)?,
                };
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&value_states)?
            }
        };

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// Apply Flash Attention if available
    #[cfg(feature = "flash-attn")]
    fn apply_flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        use candle_flash_attn::flash_attn;

        // Flash Attention expects (batch, seq_len, n_heads, head_dim)
        // Our tensors are (batch, n_heads, seq_len, head_dim)
        let q_t = q.transpose(1, 2)?;
        let k_t = k.transpose(1, 2)?;
        let v_t = v.transpose(1, 2)?;

        // Calculate softmax scale
        let scale = 1.0 / f64::sqrt(self.head_dim as f64) as f32;

        // Apply flash attention
        // causal=true for autoregressive generation (masked attention)
        let output = flash_attn(&q_t, &k_t, &v_t, scale, attention_mask.is_none())?;

        // Transpose back to (batch, n_heads, seq_len, head_dim)
        output.transpose(1, 2)
    }

    #[cfg(not(feature = "flash-attn"))]
    fn apply_flash_attention(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Flash Attention not available, return error to trigger fallback
        Err(candle_core::Error::Msg(
            "Flash Attention not enabled".to_string(),
        ))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(_mask) => None,
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

//! Vocos Decoder for Soprano TTS
//!
//! Implements the neural vocoder that converts hidden states to audio waveforms.
//! Uses ConvNeXt blocks with ISTFT head for audio synthesis.

use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};
use realfft::{ComplexToReal, RealFftPlanner};
use rustfft::num_complex::Complex32;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

/// Check if debug tensor saving is enabled
fn debug_tensors_enabled() -> bool {
    std::env::var("SOPRANO_DEBUG_TENSORS").is_ok()
}

/// Save tensor statistics (shape, min, max, mean) to stderr
fn log_tensor_stats(name: &str, tensor: &Tensor) -> Result<()> {
    let shape = tensor.dims();
    let min_val = tensor.min_all()?.to_scalar::<f32>()?;
    let max_val = tensor.max_all()?.to_scalar::<f32>()?;
    let mean_val = tensor.mean_all()?.to_scalar::<f32>()?;
    eprintln!(
        "[DEBUG] {}: shape={:?}, min={:.6}, max={:.6}, mean={:.6}",
        name, shape, min_val, max_val, mean_val
    );
    Ok(())
}

/// Save tensor as .npy file (binary format compatible with numpy)
fn save_tensor_npy(tensor: &Tensor, filename: &str) -> Result<()> {
    let path = Path::new(filename);
    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    let shape = tensor.dims();

    // Create numpy .npy format header
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {:?}}}",
        shape
    );
    let header_len = header.len();
    let padding = (16 - ((10 + header_len) % 16)) % 16;
    let total_header_len = 10 + header_len + padding;

    let mut file = File::create(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create {}: {}", filename, e)))?;

    // Write magic number
    file.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Write version (1.0)
    file.write_all(&[0x01, 0x00])
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Write header length (little-endian u16)
    file.write_all(&(total_header_len as u16).to_le_bytes())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Write header
    file.write_all(header.as_bytes())
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // Write padding
    for _ in 0..padding {
        file.write_all(&[0x20]) // space character
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    }
    file.write_all(&[0x0a]) // newline
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    // Write data as little-endian f32
    for val in data {
        file.write_all(&val.to_le_bytes())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    }

    eprintln!("[DEBUG] Saved tensor to {}", filename);
    Ok(())
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // or approximated as: 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt() as f32;
    let coeff = 0.044715f32;

    let x_cubed = x.sqr()?.mul(x)?;
    let coeff_tensor = Tensor::new(coeff, x.device())?;
    let inner = x.broadcast_add(&x_cubed.broadcast_mul(&coeff_tensor)?)?;
    let sqrt_tensor = Tensor::new(sqrt_2_over_pi, x.device())?;
    let tanh_arg = inner.broadcast_mul(&sqrt_tensor)?;
    let tanh_val = tanh_arg.tanh()?;
    let one = Tensor::ones_like(x)?;
    let half = Tensor::new(0.5f32, x.device())?;

    x.mul(&one.broadcast_add(&tanh_val)?)?.broadcast_mul(&half)
}

/// ConvNeXt Block
pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: candle_nn::Linear,
    pwconv2: candle_nn::Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    /// Create a new ConvNeXt block
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        dw_kernel_size: usize,
        layer_scale_init_value: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Depthwise convolution
        let dwconv_config = Conv1dConfig {
            padding: dw_kernel_size / 2,
            groups: dim,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let dwconv = candle_nn::conv1d(dim, dim, dw_kernel_size, dwconv_config, vb.pp("dwconv"))?;

        // Layer normalization
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;

        // Pointwise convolutions implemented as linear layers
        let pwconv1 = candle_nn::linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(intermediate_dim, dim, vb.pp("pwconv2"))?;

        // Layer scale (gamma)
        let gamma = if layer_scale_init_value > 0.0 {
            let gamma = vb.get_with_hints(
                &[dim],
                "gamma",
                candle_nn::init::Init::Const(layer_scale_init_value),
            )?;
            Some(gamma)
        } else {
            None
        };

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise convolution: (B, C, T) -> (B, C, T)
        let mut x = self.dwconv.forward(x)?;

        // Transpose for layer norm: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)?;
        x = self.norm.forward(&x)?;

        // Pointwise convolutions
        x = self.pwconv1.forward(&x)?;
        x = gelu(&x)?;
        x = self.pwconv2.forward(&x)?;

        // Apply layer scale if present
        if let Some(ref gamma) = self.gamma {
            // x is (B, T, C) here, gamma is (C). Broadcast to (1, 1, C).
            let c = gamma.dim(0)?;
            let gamma = gamma.reshape((1, 1, c))?;
            x = x.broadcast_mul(&gamma)?;
        }

        // Transpose back: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)?;

        // Residual connection
        x = residual.add(&x)?;

        Ok(x)
    }
}

/// ISTFT Head for predicting STFT complex coefficients.
///
/// Matches the Python `ISTFTHead`:
/// - out_dim = n_fft + 2
/// - padding = "center"
pub struct ISTFTHead {
    out: candle_nn::Linear,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    window: Vec<f32>,
    window_sq: Vec<f32>,
    irfft: Arc<dyn ComplexToReal<f32>>,
}

impl ISTFTHead {
    pub fn new(dim: usize, n_fft: usize, hop_length: usize, vb: VarBuilder) -> Result<Self> {
        let out_dim = n_fft + 2;
        let out = candle_nn::linear(dim, out_dim, vb.pp("out"))?;
        let win_length = n_fft;
        // Periodic Hann window matches PyTorch's default torch.hann_window(win_length)
        // The periodic window divides by win_length (not win_length-1) for exact periodicity
        let window: Vec<f32> = (0..win_length)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos())
            })
            .collect();
        let window_sq: Vec<f32> = window.iter().map(|v| v * v).collect();

        let mut planner = RealFftPlanner::<f32>::new();
        let irfft = planner.plan_fft_inverse(n_fft);

        Ok(Self {
            out,
            n_fft,
            hop_length,
            win_length,
            window,
            window_sq,
            irfft,
        })
    }

    /// Input is `(B, C, L)` (Conv1D feature map).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let debug = debug_tensors_enabled();

        // (B, C, L) -> (B, L, C)
        let x = x.transpose(1, 2)?;
        // (B, L, C) -> (B, L, out_dim)
        let x = self.out.forward(&x)?;
        // (B, L, out_dim) -> (B, out_dim, L)
        let x = x.transpose(1, 2)?;

        if debug {
            log_tensor_stats("istft_input (after linear)", &x)?;
            let _ = save_tensor_npy(&x, "rust_istft_input.npy");
        }

        let out_dim = x.dim(1)?;
        let half = out_dim / 2;
        let mag_raw = x.narrow(1, 0, half)?;
        let p = x.narrow(1, half, half)?;

        let mag = mag_raw.exp()?;
        let max_val = Tensor::new(1e2f32, mag.device())?.broadcast_as(mag.shape())?;
        let mag = mag.minimum(&max_val)?;

        if debug {
            log_tensor_stats("istft_magnitude", &mag)?;
            log_tensor_stats("istft_phase", &p)?;
            let _ = save_tensor_npy(&mag, "rust_istft_magnitude.npy");
            let _ = save_tensor_npy(&p, "rust_istft_phase.npy");
        }

        let re = p.cos()?;
        let im = p.sin()?;
        let re = mag.mul(&re)?;
        let im = mag.mul(&im)?;

        if debug {
            log_tensor_stats("istft_real_part", &re)?;
            log_tensor_stats("istft_imag_part", &im)?;
            let _ = save_tensor_npy(&re, "rust_istft_real.npy");
            let _ = save_tensor_npy(&im, "rust_istft_imag.npy");
        }

        let audio = self.istft_center(&re, &im)?;

        if debug {
            log_tensor_stats("istft_output_audio", &audio)?;
        }

        Ok(audio)
    }

    fn istft_center(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        let (b, n_freqs, n_frames) = re.dims3()?;
        let expected = self.n_fft / 2 + 1;
        if n_freqs != expected {
            return Err(candle_core::Error::Msg(format!(
                "invalid n_freqs {n_freqs}, expected {expected}"
            )));
        }

        let re_v: Vec<Vec<Vec<f32>>> = re.to_vec3()?;
        let im_v: Vec<Vec<Vec<f32>>> = im.to_vec3()?;

        let output_size = (n_frames - 1) * self.hop_length + self.win_length;

        // Use gentler trim for short sequences to preserve audio content
        // Short sequences don't have a "steady-state" middle region
        let pad = if n_frames < 5 {
            self.n_fft / 4 // 512 for short sequences (T <= 2)
        } else {
            self.n_fft / 2 // 1024 for normal sequences
        };
        let output_len = output_size.saturating_sub(2 * pad);

        let mut spec = vec![Complex32::new(0.0, 0.0); n_freqs];
        let mut time = vec![0.0f32; self.n_fft];

        let mut out_all = Vec::with_capacity(b * output_len);

        for bi in 0..b {
            let mut out = vec![0.0f32; output_size];
            let mut win_env = vec![0.0f32; output_size];

            for ti in 0..n_frames {
                for fi in 0..n_freqs {
                    let mut r = re_v[bi][fi][ti];
                    let mut m = im_v[bi][fi][ti];
                    if fi == 0 || fi + 1 == n_freqs {
                        r = 0.0;
                        m = 0.0;
                    }
                    spec[fi] = Complex32::new(r, m);
                }

                self.irfft
                    .process(&mut spec, &mut time)
                    .map_err(|e| candle_core::Error::Msg(format!("irfft failed: {e}")))?;

                // PyTorch's torch.istft with center=True does NOT apply 1/n_fft
                // normalization. It only normalizes by the window envelope.
                // The realfft output matches torch.fft.irfft with norm="backward".

                let start = ti * self.hop_length;
                for (wi, &tv) in time.iter().enumerate().take(self.win_length) {
                    let idx = start + wi;
                    if idx < output_size {
                        let w = self.window[wi];
                        // No n_fft normalization here - PyTorch handles this differently
                        out[idx] += tv * w;
                        win_env[idx] += self.window_sq[wi];
                    }
                }
            }

            for (i, v) in out.iter_mut().enumerate() {
                let denom = win_env[i];
                // Clamp envelope to prevent amplification of noise at edges
                // where window overlap is minimal
                let denom = denom.max(1e-3);
                *v /= denom;
            }

            // Apply conditional trim (gentler for short sequences)
            out_all.extend_from_slice(&out[pad..output_size.saturating_sub(pad)]);
        }

        Tensor::new(out_all, re.device())?.reshape(&[b, output_len])
    }
}

/// Vocos Backbone built with ConvNeXt blocks
pub struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
}

impl VocosBackbone {
    /// Create a new Vocos backbone
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        input_kernel_size: usize,
        dw_kernel_size: usize,
        layer_scale_init_value: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Initial embedding convolution
        let embed_config = Conv1dConfig {
            padding: input_kernel_size / 2,
            groups: 1,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let embed = candle_nn::conv1d(
            input_channels,
            dim,
            input_kernel_size,
            embed_config,
            vb.pp("embed"),
        )?;

        // Initial layer norm
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;

        // ConvNeXt blocks
        let mut convnext = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_vb = vb.pp(format!("convnext.{}", i));
            let layer_scale = if layer_scale_init_value > 0.0 {
                layer_scale_init_value
            } else {
                1.0 / (num_layers as f64).sqrt()
            };
            let block =
                ConvNeXtBlock::new(dim, intermediate_dim, dw_kernel_size, layer_scale, layer_vb)?;
            convnext.push(block);
        }

        // Final layer norm
        let final_layer_norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("final_layer_norm"))?;

        Ok(Self {
            embed,
            norm,
            convnext,
            final_layer_norm,
        })
    }
}

impl Module for VocosBackbone {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x is (B, C, L)
        let mut x = self.embed.forward(x)?;

        // Transpose for layer norm: (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)?;
        x = self.norm.forward(&x)?;
        x = x.transpose(1, 2)?; // (B, C, L)

        // Apply ConvNeXt blocks
        for block in &self.convnext {
            x = block.forward(&x)?;
        }

        // Final layer norm
        x = x.transpose(1, 2)?;
        x = self.final_layer_norm.forward(&x)?;
        x = x.transpose(1, 2)?;

        Ok(x)
    }
}

/// Complete Soprano Decoder
pub struct SopranoDecoder {
    decoder: VocosBackbone,
    head: ISTFTHead,
    upscale: usize,
    token_size: usize,
}

impl SopranoDecoder {
    /// Create a new Soprano decoder
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_input_channels: usize,
        decoder_num_layers: usize,
        decoder_dim: usize,
        decoder_intermediate_dim: usize,
        hop_length: usize,
        n_fft: usize,
        upscale: usize,
        dw_kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let decoder = VocosBackbone::new(
            num_input_channels,
            decoder_dim,
            decoder_intermediate_dim,
            decoder_num_layers,
            1, // input_kernel_size
            dw_kernel,
            1.0 / (decoder_num_layers as f64).sqrt(),
            vb.pp("decoder"),
        )?;

        let head = ISTFTHead::new(decoder_dim, n_fft, hop_length, vb.pp("head"))?;

        Ok(Self {
            decoder,
            head,
            upscale,
            token_size: 2048, // Number of samples per audio token
        })
    }

    /// Set token size (samples per token)
    pub fn set_token_size(&mut self, size: usize) {
        self.token_size = size;
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let debug = debug_tensors_enabled();

        // x is (B, C, T) where C is hidden dimension
        let dims = x.dims();
        let t = dims[2];

        if debug {
            log_tensor_stats("decoder_input", x)?;
            let _ = save_tensor_npy(x, "rust_decoder_input.npy");
        }

        // Upsample
        let target_len = self.upscale * (t - 1) + 1;
        let x = self.interpolate_1d(x, target_len)?;

        if debug {
            log_tensor_stats("upsampled", &x)?;
            let _ = save_tensor_npy(&x, "rust_upsampled.npy");
        }

        // Decode
        let x = self.decoder.forward(&x)?;

        if debug {
            log_tensor_stats("backbone_output", &x)?;
            let _ = save_tensor_npy(&x, "rust_backbone_output.npy");
        }

        // Generate audio
        let audio = self.head.forward(&x)?;

        if debug {
            log_tensor_stats("audio_output", &audio)?;
            let _ = save_tensor_npy(&audio, "rust_audio.npy");
        }

        Ok(audio)
    }

    /// 1D interpolation using linear upsampling
    fn interpolate_1d(&self, x: &Tensor, target_len: usize) -> Result<Tensor> {
        let dims = x.dims();
        let batch = dims[0];
        let channels = dims[1];
        let src_len = dims[2];

        if src_len == target_len {
            return Ok(x.clone());
        }

        // Simple linear interpolation
        let mut output = Vec::with_capacity(batch * channels * target_len);

        for b in 0..batch {
            for c in 0..channels {
                for t in 0..target_len {
                    let src_t = (t as f32) * (src_len as f32 - 1.0) / (target_len as f32 - 1.0);
                    let t0 = src_t.floor() as usize;
                    let t1 = (t0 + 1).min(src_len - 1);
                    let alpha = src_t - t0 as f32;

                    let v0 = x.i((b, c, t0))?.to_scalar::<f32>()?;
                    let v1 = x.i((b, c, t1))?.to_scalar::<f32>()?;
                    let v = v0 * (1.0 - alpha) + v1 * alpha;
                    output.push(v);
                }
            }
        }

        Tensor::new(output, x.device())?.reshape(&[batch, channels, target_len])
    }

    /// Decode hidden states to audio with proper trimming
    pub fn decode(&self, hidden_states: &Tensor, trim_tokens: usize) -> Result<Tensor> {
        // hidden_states is (T, C) or (B, T, C)
        let audio = self.forward(hidden_states)?;

        // Trim leading tokens (receptive field)
        if trim_tokens > 0 {
            let trim_samples = trim_tokens * self.token_size - self.token_size;
            let dims = audio.dims();
            let len = dims[1];
            if len > trim_samples {
                audio.narrow(1, trim_samples, len - trim_samples)
            } else {
                Ok(audio)
            }
        } else {
            Ok(audio)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;

    #[test]
    fn test_convnext_block() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let block = ConvNeXtBlock::new(64, 192, 9, 1.0, vb).unwrap();

        let input = Tensor::zeros((2, 64, 100), DType::F32, &device).unwrap();
        let output = block.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 64, 100]);
    }

    #[test]
    fn test_istft_head() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let head = ISTFTHead::new(768, 2048, 512, vb).unwrap();

        // Input: (B, C, L)
        let input = Tensor::zeros((1, 768, 10), DType::F32, &device).unwrap();
        let output = head.forward(&input).unwrap();

        // Output should be audio waveform
        assert_eq!(output.dims()[0], 1); // Batch size
        assert!(output.dims()[1] > 0); // Audio length
    }

    #[test]
    fn test_vocos_backbone() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let backbone = VocosBackbone::new(512, 768, 2304, 8, 1, 3, 0.5, vb).unwrap();

        // Input: (B, C, L)
        let input = Tensor::zeros((2, 512, 50), DType::F32, &device).unwrap();
        let output = backbone.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 768, 50]);
    }

    #[test]
    fn test_soprano_decoder() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let decoder = SopranoDecoder::new(
            512,  // num_input_channels
            8,    // decoder_num_layers
            768,  // decoder_dim
            2304, // decoder_intermediate_dim
            512,  // hop_length
            2048, // n_fft
            4,    // upscale
            3,    // dw_kernel
            vb,
        )
        .unwrap();

        // Input: (B, C, T)
        let input = Tensor::zeros((1, 512, 10), DType::F32, &device).unwrap();
        let output = decoder.forward(&input).unwrap();

        // Output should be audio
        assert_eq!(output.dims()[0], 1);
        assert!(output.dims()[1] > 0);
    }
}

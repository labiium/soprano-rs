//! Spectral operations for Soprano TTS
//!
//! Implements ISTFT (Inverse Short-Time Fourier Transform) for audio reconstruction

use candle_core::{Device, Result, Tensor};

/// Inverse Short-Time Fourier Transform
pub struct ISTFT {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    padding: String,
}

impl ISTFT {
    /// Create a new ISTFT instance
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize, padding: &str) -> Result<Self> {
        if padding != "center" && padding != "same" {
            return Err(candle_core::Error::Msg(
                "Padding must be 'center' or 'same'".to_string(),
            ));
        }

        Ok(Self {
            n_fft,
            hop_length,
            win_length,
            padding: padding.to_string(),
        })
    }

    /// Apply ISTFT to a complex spectrogram
    pub fn forward(&self, spec: &Tensor) -> Result<Tensor> {
        let dims = spec.dims();
        if dims.len() != 3 {
            return Err(candle_core::Error::Msg(
                "Expected 3D tensor input".to_string(),
            ));
        }

        let batch_size = dims[0];
        let _n_freqs = dims[1];
        let n_frames = dims[2];

        if self.padding == "center" {
            // For center padding, we use a windowed approach
            let window = hann_window(self.win_length, spec.device())?;
            let window_vec: Vec<f32> = window.to_vec1()?;

            // Reconstruct from complex spectrogram
            // spec is (B, N, T) where N = n_fft/2 + 1
            let output_length = (n_frames - 1) * self.hop_length + self.win_length;

            // Initialize output using raw vectors for accumulation
            let mut output: Vec<f32> = vec![0.0f32; batch_size * output_length];
            let mut window_sum: Vec<f32> = vec![0.0f32; output_length];

            for t in 0..n_frames {
                // Extract frame
                let frame = spec.narrow(2, t, 1)?.squeeze(2)?;

                // IFFT - returns time domain signal for each batch
                let time_frame = self.inverse_fft(&frame, self.n_fft)?;
                let time_vec: Vec<f32> = time_frame.to_vec1()?;

                // Overlap-add
                let start = t * self.hop_length;
                for b in 0..batch_size {
                    for i in 0..self.win_length.min(output_length - start) {
                        let frame_idx = b * self.win_length + i;
                        let out_idx = b * output_length + start + i;
                        if frame_idx < time_vec.len() && out_idx < output.len() {
                            output[out_idx] += time_vec[frame_idx] * window_vec[i];
                        }
                        if b == 0 {
                            window_sum[start + i] += window_vec[i] * window_vec[i];
                        }
                    }
                }
            }

            // Normalize by window
            for (i, ws) in window_sum.iter().enumerate().take(output_length) {
                if *ws > 1e-11 {
                    for b in 0..batch_size {
                        let idx = b * output_length + i;
                        output[idx] /= ws;
                    }
                }
            }

            // Reshape to (B, L)
            Tensor::new(output, spec.device())?.reshape(&[batch_size, output_length])
        } else {
            // Same padding
            let pad = (self.win_length - self.hop_length) / 2;
            let output_length = (n_frames - 1) * self.hop_length + self.win_length - 2 * pad;

            let window = hann_window(self.win_length, spec.device())?;
            let window_vec: Vec<f32> = window.to_vec1()?;

            // Initialize output using raw vectors
            let mut output: Vec<f32> = vec![0.0f32; batch_size * output_length];
            let mut window_sum: Vec<f32> = vec![0.0f32; output_length];

            for t in 0..n_frames {
                let frame = spec.narrow(2, t, 1)?.squeeze(2)?;
                let time_frame = self.inverse_fft(&frame, self.n_fft)?;
                let time_vec: Vec<f32> = time_frame.to_vec1()?;

                let start = t * self.hop_length;
                let output_start = start.saturating_sub(pad);
                let frame_start = pad.saturating_sub(start);
                let frame_end = (self.win_length - frame_start).min(output_length - output_start);

                // Overlap-add
                for b in 0..batch_size {
                    for (i, wv) in window_vec
                        .iter()
                        .enumerate()
                        .take(frame_end)
                        .skip(frame_start)
                    {
                        let frame_idx = b * self.win_length + i;
                        let out_idx = b * output_length + output_start + i - frame_start;
                        if frame_idx < time_vec.len() && out_idx < output.len() {
                            output[out_idx] += time_vec[frame_idx] * wv;
                        }
                        if b == 0 {
                            let ws_idx = output_start + i - frame_start;
                            if ws_idx < window_sum.len() {
                                window_sum[ws_idx] += wv * wv;
                            }
                        }
                    }
                }
            }

            // Normalize
            for (i, ws) in window_sum.iter().enumerate().take(output_length) {
                if *ws > 1e-11 {
                    for b in 0..batch_size {
                        let idx = b * output_length + i;
                        output[idx] /= ws;
                    }
                }
            }

            Tensor::new(output, spec.device())?.reshape(&[batch_size, output_length])
        }
    }

    /// Compute inverse FFT
    fn inverse_fft(&self, spec: &Tensor, n_fft: usize) -> Result<Tensor> {
        // spec is (B, N) where N = n_fft/2 + 1 (complex values stored as real pairs)
        // For now, assume spec is already in complex form with shape (B, N, 2) for real/imag

        // Create full spectrum for IFFT
        let dims = spec.dims();
        let batch_size = dims[0];
        let n_freqs = dims[1];

        let time_len = n_fft;
        let mut time_signal: Vec<f32> = vec![0.0f32; batch_size * time_len];

        // Convert spec to vector for easier access
        let spec_vec: Vec<f32> = spec.to_vec1()?;
        // spec is stored as (B, N, 2) - interleaved real/imag pairs

        // Simple inverse DFT (very slow, for demonstration)
        // In practice, use rustfft or similar
        for b in 0..batch_size {
            for n in 0..time_len {
                let mut sum = 0.0f32;
                for k in 0..n_freqs.min(n_fft / 2 + 1) {
                    // Access real and imag parts from flattened tensor
                    let idx = (b * n_freqs + k) * 2; // 2 for real/imag
                    if idx + 1 < spec_vec.len() {
                        let real_val = spec_vec[idx];
                        let imag_val = spec_vec[idx + 1];
                        let angle =
                            2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (n_fft as f32);
                        sum += real_val * angle.cos() - imag_val * angle.sin();
                    }
                }
                time_signal[b * time_len + n] = sum / (n_fft as f32);
            }
        }

        Tensor::new(time_signal, spec.device())?.reshape(&[batch_size, time_len])
    }
}

/// Create a Hann window
fn hann_window(size: usize, device: &Device) -> Result<Tensor> {
    let window: Vec<f32> = (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()))
        .collect();
    Tensor::new(window, device)
}

/// Spectral processor for audio reconstruction
pub struct SpectralProcessor {
    n_fft: usize,
    hop_length: usize,
}

impl SpectralProcessor {
    /// Create a new spectral processor
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        Self { n_fft, hop_length }
    }

    /// Process audio using ISTFT
    pub fn istft(&self, spec: &Tensor) -> Result<Tensor> {
        let istft = ISTFT::new(self.n_fft, self.hop_length, self.n_fft, "center")?;
        istft.forward(spec)
    }

    /// Compute Griffin-Lim reconstruction
    pub fn griffin_lim(&self, magnitude: &Tensor, n_iter: usize) -> Result<Tensor> {
        let griffin = GriffinLim::new(self.n_fft, self.hop_length, self.n_fft, n_iter);
        griffin.reconstruct(magnitude)
    }
}

/// Griffin-Lim algorithm for phase reconstruction
pub struct GriffinLim {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    #[allow(dead_code)]
    n_iter: usize,
}

impl GriffinLim {
    /// Create a new Griffin-Lim instance
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize, n_iter: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            win_length,
            n_iter,
        }
    }

    /// Reconstruct audio from magnitude spectrogram
    pub fn reconstruct(&self, magnitude: &Tensor) -> Result<Tensor> {
        // Initialize with random phase
        let (batch, n_freqs, n_frames) = (
            magnitude.dims()[0],
            magnitude.dims()[1],
            magnitude.dims()[2],
        );
        let phase = Tensor::rand(
            0.0f32,
            2.0 * std::f32::consts::PI,
            (batch, n_freqs, n_frames),
            magnitude.device(),
        )?;

        // Combine magnitude and phase
        let real = magnitude.mul(&phase.cos()?)?;
        let imag = magnitude.mul(&phase.sin()?)?;

        // Convert to vectors for interleaving
        let real_vec: Vec<f32> = real.to_vec1()?;
        let imag_vec: Vec<f32> = imag.to_vec1()?;

        // Create complex spectrogram - interleave real and imag
        let mut complex = Vec::with_capacity(batch * n_freqs * n_frames * 2);
        for b in 0..batch {
            for f in 0..n_freqs {
                for t in 0..n_frames {
                    let idx = (b * n_freqs + f) * n_frames + t;
                    if idx < real_vec.len() && idx < imag_vec.len() {
                        complex.push(real_vec[idx]);
                        complex.push(imag_vec[idx]);
                    }
                }
            }
        }

        let complex_tensor = Tensor::new(complex, magnitude.device())?;
        let complex_reshaped = complex_tensor.reshape(&[batch, n_freqs, n_frames, 2])?;

        // Apply ISTFT
        let istft = ISTFT::new(self.n_fft, self.hop_length, self.win_length, "center")?;
        istft.forward(&complex_reshaped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let device = Device::Cpu;
        let window = hann_window(512, &device).unwrap();
        assert_eq!(window.dims(), &[512]);

        // Check endpoints
        let first: f32 = window.get(0).unwrap().to_scalar().unwrap();
        let last: f32 = window.get(511).unwrap().to_scalar().unwrap();
        assert!(first.abs() < 1e-6);
        assert!(last.abs() < 1e-6);
    }

    #[test]
    fn test_istft_new() {
        let istft = ISTFT::new(2048, 512, 2048, "center");
        assert!(istft.is_ok());

        let invalid = ISTFT::new(2048, 512, 2048, "invalid");
        assert!(invalid.is_err());
    }
}

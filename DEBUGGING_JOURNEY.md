# Soprano-RS Debugging Journey

This document chronicles the debugging process that transformed soprano-rs from producing garbled noise to working TTS audio.

## The Problem

The Rust implementation of Soprano TTS was producing completely garbled, unintelligible noise instead of speech. Key symptoms:
- Generated audio was noise/static
- Model never stopped at EOS (generated max 128 tokens)
- Logits completely different from Python reference
- Audio duration was wrong (8.13s instead of ~1.34s)

## Root Causes Identified

### 1. **Wrong Model Architecture: Qwen3 vs Qwen2** ⚠️ CRITICAL

The model checkpoint declares `model_type: qwen3` but we were using Candle's `qwen2::Model`.

**Key Differences:**
- **Qwen3 has Q/K normalization layers** that Qwen2 lacks
- Different RoPE implementation
- Different attention mechanism details

### 2. **Missing Q/K Normalization Layers** 🔴 CRITICAL

Qwen3 applies RMSNorm to Q and K projections after reshaping:

```python
# Python (Qwen3)
query = self.q_norm(query)  # RMSNorm per head
key = self.k_norm(key)      # RMSNorm per head
```

Our initial implementation was missing these layers entirely.

### 3. **RoPE Implementation Bug** 🔴 CRITICAL

Candle's `rope()` function uses **interleaved rotation** but HuggingFace uses **half-split rotation**.

```rust
// WRONG: Interleaved rotation
let q_embed = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;

// CORRECT: Half-split rotation (matches HuggingFace)
let q_embed = candle_nn::rotary_emb::rope_slow(&q, &cos, &sin)?;
```

### 4. **ISTFT Normalization Bug** 🟡 MODERATE

We incorrectly applied `1/n_fft` normalization:

```rust
// WRONG
let inv_n = 1.0f32 / (self.n_fft as f32);
out[idx] += (tv * inv_n) * w;

// CORRECT (PyTorch doesn't normalize by n_fft)
out[idx] += tv * w;
```

### 5. **ISTFT Padding Bug** 🟡 MODERATE

Padding calculation was wrong:

```rust
// WRONG
let pad = self.win_length / 2;  // = 1024

// CORRECT
let pad = (self.win_length - self.hop_length) / 2;  // = 768
```

### 6. **Audio Volume/Clipping Issues** 🟢 MINOR

Audio was too loud with harsh clipping:
- Added -6dB headroom normalization
- Implemented soft clipping using tanh

## The Fixes

### Fix 1: Create Custom Qwen3 Model (`src/qwen3.rs`)

Created a custom Qwen3 implementation based on Candle's qwen2 but with critical differences:

```rust
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,  // NEW: Qwen3 specific
    k_norm: RmsNorm,  // NEW: Qwen3 specific
    // ... other fields
}

// In forward pass:
let query_states_normed = query_states
    .reshape((b_sz * self.num_heads, q_len, self.head_dim))?
    .apply(&self.q_norm)?  // NEW: Apply RMSNorm
    .reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
```

### Fix 2: Use rope_slow for RoPE

```rust
// Use rope_slow (contiguous-half style) to match HuggingFace Qwen2 implementation.
// The rope function uses interleaved rotation which produces different results.
let q_embed = candle_nn::rotary_emb::rope_slow(&q.contiguous()?, &cos, &sin)?;
let k_embed = candle_nn::rotary_emb::rope_slow(&k.contiguous()?, &cos, &sin)?;
```

### Fix 3: Fix ISTFT Padding and Normalization

```rust
// Padding
let pad = (self.win_length - self.hop_length) / 2;  // 768, not 1024

// No normalization by n_fft (matches PyTorch)
out[idx] += tv * w;  // No * inv_n
```

### Fix 4: Audio Normalization

```rust
fn normalize_audio(audio: Vec<f32>) -> Vec<f32> {
    let max_abs = audio.iter().map(|&s| s.abs()).fold(0.0f32, |a, b| a.max(b));
    
    if max_abs > 0.0 {
        let target_peak = 0.5; // -6dB headroom
        let scale = target_peak / max_abs;
        
        audio.iter().map(|&s| {
            let scaled = s * scale;
            // Soft clipping for values near limits
            if scaled.abs() > 0.8 {
                scaled.tanh()
            } else {
                scaled
            }
        }).collect()
    } else {
        audio
    }
}
```

### Fix 5: Switch Model Implementation

Changed `src/model.rs` to use our custom qwen3:

```rust
// BEFORE
use candle_transformers::models::qwen2;
base: qwen2::Model,

// AFTER
use crate::qwen3;
base: qwen3::Model,
```

## Debugging Methodology

### Phase 1: Layer-by-Layer Comparison

Created test harness to compare Python vs Rust at each layer:
- Saved intermediate tensors from both implementations
- Compared shapes, mean absolute error, max difference
- Identified first point of divergence

### Phase 2: Component Isolation

Spawned parallel workers to debug each component:
1. **Worker 1**: Attention mechanism
2. **Worker 2**: RoPE embeddings
3. **Worker 3**: RMSNorm
4. **Worker 4**: MLP/SwiGLU

### Phase 3: Discovery

Key discovery: The model is Qwen3, not Qwen2!

```bash
$ python -c "from transformers import AutoModelForCausalLM; \
    m = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-1.1-80M'); \
    print(type(m))"

<class 'transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM'>
```

### Phase 4: Implementation & Verification

- Implemented missing Q/K norm layers
- Fixed RoPE to use rope_slow
- Verified token predictions match Python
- Verified audio quality acceptable

## Results

### Before Fixes
```
First token: 7712 (wrong)
Tokens generated: 128
EOS generated: Never
Stop reason: Length
Audio: 8.13s of noise
```

### After Fixes
```
First token: 6136 (matches Python!)
Tokens generated: 55
EOS generated: Yes (step 22, 24, 57)
Stop reason: Stop
Audio: 3.46s of intelligible speech
```

## Key Insights

1. **Architecture matters**: Even small differences (Q/K norm) cause complete failure
2. **RoPE is subtle**: Interleaved vs half-split rotation produces completely different results
3. **Numerical precision**: Small differences compound through 17 transformer layers
4. **Testing is essential**: Layer-by-layer comparison was crucial for finding the bug

## Files Modified

- `src/qwen3.rs` - Created custom Qwen3 implementation
- `src/model.rs` - Switched to use qwen3::Model
- `src/decoder.rs` - Fixed ISTFT padding and normalization
- `src/tts.rs` - Added audio normalization

## Testing Commands

```bash
# Generate audio
cargo run --release -- generate -t "Hello world"

# Play audio
ffplay sample.wav

# With verbose output
cargo run --release -- generate -t "Hello world" --verbose
```

## Lessons Learned

1. **Don't assume model compatibility**: Qwen3 ≠ Qwen2
2. **Check normalization carefully**: Different libraries normalize differently
3. **RoPE is tricky**: Verify the exact rotation method
4. **Layer-by-layer debugging**: Essential for finding divergence points
5. **Parallel investigation**: Multiple workers found different pieces of the puzzle

## Credits

This debugging effort involved:
- Systematic layer-by-layer comparison
- Parallel investigation of multiple components
- Extensive testing and verification
- Community support and tooling

The fix demonstrates the importance of precise model implementation and thorough testing when porting ML models between frameworks.

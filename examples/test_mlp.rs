//! Test MLP/SwiGLU implementation against HuggingFace reference
//!
//! SwiGLU formula:
//!   gate = silu(x @ W_gate)
//!   up = x @ W_up
//!   hidden = gate * up  // Element-wise multiply
//!   output = hidden @ W_down
//!
//! Where silu(x) = x * sigmoid(x)

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Linear, Module};

/// MLP implementation matching qwen3.rs
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
        act_fn: Activation,
    ) -> Result<Self> {
        let gate_proj = Linear::new(gate_weight, None);
        let up_proj = Linear::new(up_weight, None);
        let down_proj = Linear::new(down_weight, None);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

fn main() -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("Candle MLP/SwiGLU Implementation Test");
    println!("{}", "=".repeat(70));

    let _device = Device::Cpu;

    println!("\n1. Loading weights from Python...");

    let gate_proj_weight =
        Tensor::read_npy("/tmp/mlp_test/gate_proj_weight.npy")?.to_dtype(DType::F32)?;
    let up_proj_weight =
        Tensor::read_npy("/tmp/mlp_test/up_proj_weight.npy")?.to_dtype(DType::F32)?;
    let down_proj_weight =
        Tensor::read_npy("/tmp/mlp_test/down_proj_weight.npy")?.to_dtype(DType::F32)?;
    let test_input = Tensor::read_npy("/tmp/mlp_test/test_input.npy")?.to_dtype(DType::F32)?;
    let expected_output =
        Tensor::read_npy("/tmp/mlp_test/expected_output.npy")?.to_dtype(DType::F32)?;

    println!("   gate_proj weight shape: {:?}", gate_proj_weight.dims());
    println!("   up_proj weight shape: {:?}", up_proj_weight.dims());
    println!("   down_proj weight shape: {:?}", down_proj_weight.dims());
    println!("   test input shape: {:?}", test_input.dims());
    println!("   expected output shape: {:?}", expected_output.dims());

    println!("\n2. Creating MLP with SiLU activation...");
    let mlp = Mlp::new(
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
        Activation::Silu,
    )?;

    println!("\n3. Running forward pass...");
    let output = mlp.forward(&test_input)?;
    println!("   output shape: {:?}", output.dims());

    println!("\n4. Comparing with HuggingFace output...");

    let diff = (&output - &expected_output)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!("   Max absolute diff: {:.8}", max_diff);
    println!("   Mean absolute diff: {:.8}", mean_diff);

    let output_flat = output.flatten_all()?;
    let expected_flat = expected_output.flatten_all()?;

    println!("\n5. Sample output values (first 5):");
    for i in 0..5.min(output_flat.dims()[0]) {
        let candle_val = output_flat.i(i)?.to_scalar::<f32>()?;
        let python_val = expected_flat.i(i)?.to_scalar::<f32>()?;
        println!(
            "   [{}] Candle: {:12.8}, Python: {:12.8}, Diff: {:12.8}",
            i,
            candle_val,
            python_val,
            (candle_val - python_val).abs()
        );
    }

    println!("\n6. Test Result:");
    let tolerance = 1e-4;
    if max_diff < tolerance {
        println!(
            "   PASS: Outputs match within tolerance ({:.0e})",
            tolerance
        );
    } else {
        println!("   FAIL: Outputs differ by more than {:.0e}", tolerance);
        println!("   This indicates a bug in the MLP implementation!");
    }

    println!("\n7. Testing intermediate values...");

    let gate_out = test_input.apply(&mlp.gate_proj)?;
    let gate_activated = gate_out.apply(&Activation::Silu)?;

    println!("   gate_proj output shape: {:?}", gate_out.dims());
    println!("   after SiLU shape: {:?}", gate_activated.dims());

    let gate_flat = gate_out.flatten_all()?;
    let silu_flat = gate_activated.flatten_all()?;

    println!("\n   SiLU verification (first 5 values):");
    for i in 0..5.min(gate_flat.dims()[0]) {
        let x = gate_flat.i(i)?.to_scalar::<f32>()?;
        let silu_x = silu_flat.i(i)?.to_scalar::<f32>()?;
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let expected_silu = x * sigmoid_x;
        println!(
            "   x={:8.4}, silu(x)={:8.4}, expected={:8.4}, match={}",
            x,
            silu_x,
            expected_silu,
            (silu_x - expected_silu).abs() < 1e-5
        );
    }

    println!("\n{}", "=".repeat(70));
    println!("Test Complete!");
    println!("{}", "=".repeat(70));

    Ok(())
}

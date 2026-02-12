//! Test to compare MLP/SwiGLU computation with Python

use candle_core::{Device, IndexOp, Result, Tensor};

#[test]
fn test_mlp_silu_computation() -> Result<()> {
    let device = Device::Cpu;

    println!("MLP/SwiGLU Rust Implementation Test");

    // Load weights saved from Python
    let gate_weight = Tensor::read_npy("/tmp/mlp_test/gate_proj_weight.npy")?.to_device(&device)?;
    let up_weight = Tensor::read_npy("/tmp/mlp_test/up_proj_weight.npy")?.to_device(&device)?;
    let down_weight = Tensor::read_npy("/tmp/mlp_test/down_proj_weight.npy")?.to_device(&device)?;
    let test_input = Tensor::read_npy("/tmp/mlp_test/test_input.npy")?.to_device(&device)?;
    let expected_output =
        Tensor::read_npy("/tmp/mlp_test/expected_output.npy")?.to_device(&device)?;

    println!("Loaded weights from Python:");
    println!("  gate weight shape: {:?}", gate_weight.shape());
    println!("  up weight shape: {:?}", up_weight.shape());
    println!("  down weight shape: {:?}", down_weight.shape());
    println!("  test input shape: {:?}", test_input.shape());

    // Manual SwiGLU computation matching Candle's implementation
    println!("Manual SwiGLU computation:");

    // Debug shapes
    let gate_t = gate_weight.transpose(0, 1)?;
    println!("  gate_weight shape: {:?}", gate_weight.shape());
    println!("  gate_weight.T shape: {:?}", gate_t.shape());
    println!("  test_input shape: {:?}", test_input.shape());

    // Step 1: gate_proj (x @ W_gate.T)
    // Reshape input to 2D for matmul, then reshape back
    let (b, seq, hidden) = test_input.dims3()?;
    let input_2d = test_input.reshape((b * seq, hidden))?;
    let gate_proj_2d = input_2d.matmul(&gate_t)?;
    let gate_proj_out = gate_proj_2d.reshape((b, seq, 2304))?;
    println!("  gate_proj output shape: {:?}", gate_proj_out.shape());

    // Step 2: Apply SiLU to gate
    let gate_activated = candle_nn::ops::silu(&gate_proj_out)?;
    println!("  after SiLU shape: {:?}", gate_activated.shape());

    // Step 3: up_proj (x @ W_up.T)
    let up_proj_2d = input_2d.matmul(&up_weight.transpose(0, 1)?)?;
    let up_proj_out = up_proj_2d.reshape((b, seq, 2304))?;
    println!("  up_proj output shape: {:?}", up_proj_out.shape());

    // Step 4: Element-wise multiply
    let hidden = (gate_activated * up_proj_out)?;
    println!("  hidden (gate * up) shape: {:?}", hidden.shape());

    // Step 5: down_proj (hidden @ W_down.T)
    // hidden is [b, seq, 2304], down_weight is [512, 2304]
    // So hidden @ down_weight.T = [b, seq, 2304] @ [2304, 512] = [b, seq, 512]
    let hidden_2d = hidden.reshape((b * seq, 2304))?;
    let output_2d = hidden_2d.matmul(&down_weight.transpose(0, 1)?)?;
    let output_manual = output_2d.reshape((b, seq, 512))?;
    println!("  down_proj output shape: {:?}", output_manual.shape());

    // Compare with expected
    println!("Comparison with Python expected output:");

    let diff = (&output_manual - &expected_output)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    let mean_diff = diff.mean_all()?.to_scalar::<f32>()?;

    println!("  Max absolute diff: {:.8}", max_diff);
    println!("  Mean absolute diff: {:.8}", mean_diff);

    // Print sample values
    let rust_sample: Vec<f32> = output_manual.i((0, 0, ..5))?.to_vec1()?;
    let python_sample: Vec<f32> = expected_output.i((0, 0, ..5))?.to_vec1()?;
    println!("Sample output values (first 5 of first token):");
    println!("  Python: {:?}", python_sample);
    println!("  Rust:   {:?}", rust_sample);

    // Check if outputs match
    let match_threshold = 1e-4;
    let outputs_match = max_diff < match_threshold;
    println!("Outputs match (< {}): {}", match_threshold, outputs_match);

    assert!(
        outputs_match,
        "MLP outputs don't match! Max diff: {}",
        max_diff
    );

    Ok(())
}

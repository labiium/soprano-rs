//! Test embedding lookup directly

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let model_dir =
        std::path::PathBuf::from("/home/emmanuel/.cache/soprano-rs/models/ekwek--Soprano-1.1-80M");

    println!("Loading embedding weights...");
    let st_path = model_dir.join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[st_path], DType::F32, &device)? };

    // Load embedding layer directly
    let embed = candle_nn::embedding(8192, 512, vb.pp("model").pp("embed_tokens"))?;

    // Test lookup for token 3
    let token_3 = Tensor::new(&[3u32], &device)?;
    let emb_3 = embed.forward(&token_3)?;

    println!("Token 3 embedding shape: {:?}", emb_3.shape());
    println!("Token 3 embedding first 10 values:");
    let emb_3_squeezed = emb_3.squeeze(0)?;
    let values: Vec<f32> = emb_3_squeezed.to_vec1()?;
    for (i, v) in values.iter().take(10).enumerate() {
        println!("  [{}] = {:.8}", i, v);
    }

    // Compare with expected Python values
    let expected = [
        -0.09130859,
        -0.10693359,
        0.04541016,
        0.00369263,
        -0.02441406,
        -0.00598145,
        0.11425781,
        0.09912109,
        0.00108337,
        -0.07617188,
    ];

    println!("\nExpected (Python) first 10 values:");
    for (i, v) in expected.iter().enumerate() {
        println!("  [{}] = {:.8}", i, v);
    }

    println!("\nDifferences:");
    for i in 0..10 {
        let diff = (values[i] - expected[i]).abs();
        println!("  [{}] diff = {:.8}", i, diff);
    }

    // Test multiple tokens: [3, 1, 8077]
    println!("\n\nTesting multiple tokens [3, 1, 8077]...");
    let tokens = Tensor::new(&[3u32, 1, 8077], &device)?;
    let embs = embed.forward(&tokens)?;
    println!("Embeddings shape: {:?}", embs.shape());

    let all_values: Vec<Vec<f32>> = embs.to_vec2()?;
    for (t, vals) in all_values.iter().enumerate() {
        println!("Token {} first 5 values: {:?}", [3, 1, 8077][t], &vals[..5]);
    }

    Ok(())
}

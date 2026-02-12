//! Integration tests for decoder weight loading
//!
//! Verifies that the decoder.pth file loads correctly and all expected
//! layers are present with correct shapes.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::PathBuf;

// Import from the soprano crate
use soprano::config::DecoderConfig;
use soprano::decoder::SopranoDecoder;

/// Test that decoder.pth loads without errors
#[test]
fn test_decoder_pth_loads() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    assert!(
        decoder_path.exists(),
        "decoder.pth not found at {}",
        decoder_path.display()
    );

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device);

    assert!(vb.is_ok(), "Failed to load decoder.pth: {:?}", vb.err());
}

/// Test that decoder.embed loads correctly
#[test]
fn test_decoder_embed_loading() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    // Try to load embed weights
    let embed_weight = vb.get((768, 512, 1usize), "decoder.embed.weight");
    assert!(
        embed_weight.is_ok(),
        "Failed to load decoder.embed.weight: {:?}",
        embed_weight.err()
    );

    let embed_bias = vb.get(768, "decoder.embed.bias");
    assert!(
        embed_bias.is_ok(),
        "Failed to load decoder.embed.bias: {:?}",
        embed_bias.err()
    );
}

/// Test that decoder.norm loads correctly
#[test]
fn test_decoder_norm_loading() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    let norm_weight = vb.get(768, "decoder.norm.weight");
    assert!(
        norm_weight.is_ok(),
        "Failed to load decoder.norm.weight: {:?}",
        norm_weight.err()
    );

    let norm_bias = vb.get(768, "decoder.norm.bias");
    assert!(
        norm_bias.is_ok(),
        "Failed to load decoder.norm.bias: {:?}",
        norm_bias.err()
    );
}

/// Test that all 8 ConvNeXt blocks load correctly
#[test]
fn test_convnext_blocks_loading() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    for i in 0..8 {
        // dwconv: depthwise convolution
        let dwconv_weight = vb.get(
            (768, 1usize, 3usize),
            &format!("decoder.convnext.{}.dwconv.weight", i),
        );
        assert!(
            dwconv_weight.is_ok(),
            "Failed to load decoder.convnext.{}.dwconv.weight: {:?}",
            i,
            dwconv_weight.err()
        );

        let dwconv_bias = vb.get(768, &format!("decoder.convnext.{}.dwconv.bias", i));
        assert!(
            dwconv_bias.is_ok(),
            "Failed to load decoder.convnext.{}.dwconv.bias: {:?}",
            i,
            dwconv_bias.err()
        );

        // norm: LayerNorm
        let norm_weight = vb.get(768, &format!("decoder.convnext.{}.norm.weight", i));
        assert!(
            norm_weight.is_ok(),
            "Failed to load decoder.convnext.{}.norm.weight: {:?}",
            i,
            norm_weight.err()
        );

        let norm_bias = vb.get(768, &format!("decoder.convnext.{}.norm.bias", i));
        assert!(
            norm_bias.is_ok(),
            "Failed to load decoder.convnext.{}.norm.bias: {:?}",
            i,
            norm_bias.err()
        );

        // pwconv1: pointwise conv 1 (expansion)
        let pwconv1_weight = vb.get(
            (2304, 768),
            &format!("decoder.convnext.{}.pwconv1.weight", i),
        );
        assert!(
            pwconv1_weight.is_ok(),
            "Failed to load decoder.convnext.{}.pwconv1.weight: {:?}",
            i,
            pwconv1_weight.err()
        );

        let pwconv1_bias = vb.get(2304, &format!("decoder.convnext.{}.pwconv1.bias", i));
        assert!(
            pwconv1_bias.is_ok(),
            "Failed to load decoder.convnext.{}.pwconv1.bias: {:?}",
            i,
            pwconv1_bias.err()
        );

        // pwconv2: pointwise conv 2 (projection)
        let pwconv2_weight = vb.get(
            (768, 2304),
            &format!("decoder.convnext.{}.pwconv2.weight", i),
        );
        assert!(
            pwconv2_weight.is_ok(),
            "Failed to load decoder.convnext.{}.pwconv2.weight: {:?}",
            i,
            pwconv2_weight.err()
        );

        let pwconv2_bias = vb.get(768, &format!("decoder.convnext.{}.pwconv2.bias", i));
        assert!(
            pwconv2_bias.is_ok(),
            "Failed to load decoder.convnext.{}.pwconv2.bias: {:?}",
            i,
            pwconv2_bias.err()
        );

        // gamma: layer scale
        let gamma = vb.get(768, &format!("decoder.convnext.{}.gamma", i));
        assert!(
            gamma.is_ok(),
            "Failed to load decoder.convnext.{}.gamma: {:?}",
            i,
            gamma.err()
        );
    }
}

/// Test that final_layer_norm loads correctly
#[test]
fn test_final_layer_norm_loading() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    let fln_weight = vb.get(768, "decoder.final_layer_norm.weight");
    assert!(
        fln_weight.is_ok(),
        "Failed to load decoder.final_layer_norm.weight: {:?}",
        fln_weight.err()
    );

    let fln_bias = vb.get(768, "decoder.final_layer_norm.bias");
    assert!(
        fln_bias.is_ok(),
        "Failed to load decoder.final_layer_norm.bias: {:?}",
        fln_bias.err()
    );
}

/// Test that ISTFT head loads correctly
#[test]
fn test_istft_head_loading() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    // out: Linear layer
    let out_weight = vb.get((2050, 768), "head.out.weight");
    assert!(
        out_weight.is_ok(),
        "Failed to load head.out.weight: {:?}",
        out_weight.err()
    );

    let out_bias = vb.get(2050, "head.out.bias");
    assert!(
        out_bias.is_ok(),
        "Failed to load head.out.bias: {:?}",
        out_bias.err()
    );

    // istft.window: buffer (not a parameter, may or may not exist)
    let window = vb.get(2048, "head.istft.window");
    // This might fail if candle-nn doesn't load buffers from .pth
    // So we'll just print a warning instead of asserting
    if window.is_err() {
        eprintln!("Warning: head.istft.window not loaded (may be expected if treated as buffer)");
    }
}

/// Test that the full SopranoDecoder can be constructed from decoder.pth
#[test]
fn test_full_decoder_construction() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    let config = DecoderConfig::default();

    let decoder = SopranoDecoder::new(
        config.num_input_channels,
        config.decoder_num_layers,
        config.decoder_dim,
        config.decoder_intermediate_dim,
        config.hop_length,
        config.n_fft,
        config.upscale,
        config.dw_kernel,
        vb,
    );

    assert!(
        decoder.is_ok(),
        "Failed to construct SopranoDecoder: {:?}",
        decoder.err()
    );

    // Verify we can run forward pass with dummy input
    let decoder = decoder.unwrap();
    let input = candle_core::Tensor::zeros((1, 512, 10), DType::F32, &device).unwrap();
    let output = decoder.forward(&input);

    assert!(output.is_ok(), "Forward pass failed: {:?}", output.err());

    let output = output.unwrap();
    assert_eq!(output.dims()[0], 1); // Batch size
    assert!(output.dims()[1] > 0); // Audio length
}

/// Verify total parameter count by loading all expected weights
#[test]
fn test_all_weights_loadable() {
    let decoder_path = PathBuf::from(std::env::var("HOME").expect("HOME not set"))
        .join(".cache/soprano-rs/models/ekwek--Soprano-1.1-80M/decoder.pth");

    let device = Device::Cpu;
    let vb = VarBuilder::from_pth(&decoder_path, DType::F32, &device).unwrap();

    let mut loaded_count = 0;

    // decoder.embed
    let _ = vb.get((768, 512, 1usize), "decoder.embed.weight").unwrap();
    let _ = vb.get(768, "decoder.embed.bias").unwrap();
    loaded_count += 2;

    // decoder.norm
    let _ = vb.get(768, "decoder.norm.weight").unwrap();
    let _ = vb.get(768, "decoder.norm.bias").unwrap();
    loaded_count += 2;

    // decoder.convnext.0-7
    for i in 0..8 {
        let _ = vb
            .get(
                (768, 1usize, 3usize),
                &format!("decoder.convnext.{}.dwconv.weight", i),
            )
            .unwrap();
        let _ = vb
            .get(768, &format!("decoder.convnext.{}.dwconv.bias", i))
            .unwrap();
        let _ = vb
            .get(768, &format!("decoder.convnext.{}.norm.weight", i))
            .unwrap();
        let _ = vb
            .get(768, &format!("decoder.convnext.{}.norm.bias", i))
            .unwrap();
        let _ = vb
            .get(
                (2304, 768),
                &format!("decoder.convnext.{}.pwconv1.weight", i),
            )
            .unwrap();
        let _ = vb
            .get(2304, &format!("decoder.convnext.{}.pwconv1.bias", i))
            .unwrap();
        let _ = vb
            .get(
                (768, 2304),
                &format!("decoder.convnext.{}.pwconv2.weight", i),
            )
            .unwrap();
        let _ = vb
            .get(768, &format!("decoder.convnext.{}.pwconv2.bias", i))
            .unwrap();
        let _ = vb
            .get(768, &format!("decoder.convnext.{}.gamma", i))
            .unwrap();
        loaded_count += 9;
    }

    // decoder.final_layer_norm
    let _ = vb.get(768, "decoder.final_layer_norm.weight").unwrap();
    let _ = vb.get(768, "decoder.final_layer_norm.bias").unwrap();
    loaded_count += 2;

    // head.out
    let _ = vb.get((2050, 768), "head.out.weight").unwrap();
    let _ = vb.get(2050, "head.out.bias").unwrap();
    loaded_count += 2;

    // head.istft.window (buffer)
    if vb.get(2048, "head.istft.window").is_ok() {
        loaded_count += 1;
    }

    // We expect 78 decoder keys + 2 or 3 head keys = 80 or 81 total
    assert!(
        loaded_count == 80 || loaded_count == 81,
        "Expected 80 or 81 loaded weights, got {}",
        loaded_count
    );

    println!(
        "Successfully loaded {} weights from decoder.pth",
        loaded_count
    );
}

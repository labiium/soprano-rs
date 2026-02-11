//! Test Python backend integration

use candle_core::Device;
use soprano_tts::config::GenerationConfig;
use soprano_tts::python_backend::PythonTtsBackend;

fn main() -> anyhow::Result<()> {
    println!("Testing Python backend...");

    let device = Device::Cpu;
    let mut backend = PythonTtsBackend::new(device);

    let config = GenerationConfig {
        max_new_tokens: 128,
        temperature: 0.0,
        top_p: 0.95,
        repetition_penalty: 1.2,
        min_new_tokens: 0,
    };

    println!("Generating audio for 'Hello world'...");
    let (audio, duration) = backend.generate("Hello world", &config)?;

    println!("Success!");
    println!("  Audio shape: {:?}", audio.shape());
    println!("  Duration: {:.2}s", duration);

    // Save audio to file
    let audio_vec: Vec<f32> = audio.to_vec1()?;

    // Convert to i16 and save as raw PCM
    let audio_i16: Vec<i16> = audio_vec.iter().map(|&x| (x * 32767.0) as i16).collect();

    std::fs::write(
        "/tmp/test_python_backend.raw",
        audio_i16
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;

    println!("  Saved to /tmp/test_python_backend.raw");
    println!("\nTo play: ffplay -f s16le -ar 32000 -ac 1 /tmp/test_python_backend.raw");

    Ok(())
}

//! Model downloading and loading utilities for Soprano TTS
//!
//! This module provides functionality to:
//! - Download models from HuggingFace Hub
//! - Load quantized model weights (GGUF format)
//! - Load SafeTensors models (regular transformer models)
//! - Load decoder weights (Safetensors/PyTorch format)
//! - Convert between formats as needed
//! - Verify checksums for downloaded files

use candle_core::{Device, Error as CandleError, Tensor};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, warn};

#[cfg(feature = "model-download")]
use indicatif::{ProgressBar, ProgressStyle};

/// Errors that can occur during model loading
#[derive(Error, Debug)]
pub enum ModelLoaderError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] CandleError),

    #[error("Walkdir error: {0}")]
    Walkdir(String),

    #[cfg(feature = "model-download")]
    #[error("Zip error: {0}")]
    Zip(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Checksum mismatch for {file}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Model file not found: {0}")]
    NotFound(String),

    #[error("HuggingFace error: {0}")]
    HuggingFace(String),

    #[error("Download interrupted")]
    DownloadInterrupted,

    #[error("Other error: {0}")]
    Other(String),
}

impl From<walkdir::Error> for ModelLoaderError {
    fn from(e: walkdir::Error) -> Self {
        ModelLoaderError::Walkdir(e.to_string())
    }
}

#[cfg(feature = "model-download")]
impl From<zip::result::ZipError> for ModelLoaderError {
    fn from(e: zip::result::ZipError) -> Self {
        ModelLoaderError::Zip(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ModelLoaderError>;

/// Metadata for a model file
#[derive(Debug, Clone)]
pub struct ModelFileInfo {
    pub filename: String,
    pub url: String,
    pub size: Option<u64>,
    pub checksum: Option<String>,
}

/// Supported model weight formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    /// GGUF quantized format
    Gguf,
    /// SafeTensors format
    SafeTensors,
    /// PyTorch format (requires conversion)
    PyTorch,
    /// ONNX format
    Onnx,
}

impl ModelFormat {
    /// Detect format from file extension
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Ok(ModelFormat::Gguf),
            Some("safetensors") => Ok(ModelFormat::SafeTensors),
            Some("pth") | Some("pt") => Ok(ModelFormat::PyTorch),
            Some("onnx") => Ok(ModelFormat::Onnx),
            _ => Err(ModelLoaderError::InvalidFormat(
                format!("Unknown file format: {:?}", path),
            )),
        }
    }

    /// Get the format name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFormat::Gguf => "GGUF",
            ModelFormat::SafeTensors => "SafeTensors",
            ModelFormat::PyTorch => "PyTorch",
            ModelFormat::Onnx => "ONNX",
        }
    }
}

/// Model verification result
#[derive(Debug, Clone)]
pub struct ModelVerification {
    pub format: ModelFormat,
    pub is_valid: bool,
    pub files_found: Vec<String>,
    pub errors: Vec<String>,
}

/// Configuration for model downloading
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// HuggingFace API endpoint
    pub hf_endpoint: String,
    /// Whether to show progress bars
    pub show_progress: bool,
    /// Chunk size for downloading (bytes)
    pub chunk_size: usize,
    /// Number of retry attempts
    pub max_retries: u32,
    /// Timeout for requests (seconds)
    pub timeout_secs: u64,
    /// Whether to verify checksums
    pub verify_checksums: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            hf_endpoint: "https://huggingface.co".to_string(),
            show_progress: true,
            chunk_size: 8192,
            max_retries: 3,
            timeout_secs: 300,
            verify_checksums: true,
        }
    }
}

/// Cache management for downloaded models
pub struct ModelCache {
    base_path: PathBuf,
}

impl ModelCache {
    /// Create a new model cache at the specified path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }

    /// Get the default cache directory
    pub fn default_cache_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| ModelLoaderError::Other("Could not find cache directory".to_string()))?
            .join("soprano-rs")
            .join("models");
        Ok(cache_dir)
    }

    /// Get path for a specific model
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        let sanitized_id = model_id.replace('/', "--");
        self.base_path.join(sanitized_id)
    }

    /// Check if a model exists in cache
    pub fn is_cached(&self, model_id: &str, files: &[&str]) -> bool {
        let model_path = self.model_path(model_id);
        files.iter().all(|file| model_path.join(file).exists())
    }

    /// Get cache size for a model (in bytes)
    pub fn get_cache_size(&self, model_id: &str) -> Result<u64> {
        let model_path = self.model_path(model_id);
        if !model_path.exists() {
            return Ok(0);
        }

        let mut total_size = 0u64;
        for entry in walkdir::WalkDir::new(model_path) {
            let entry = entry.map_err(|e| ModelLoaderError::Io(e.into()))?;
            if entry.file_type().is_file() {
                total_size += entry.metadata()?.len();
            }
        }
        Ok(total_size)
    }

    /// Clean up old models
    pub fn cleanup_old_models(&self, keep_model_ids: &[&str], max_age_days: u64) -> Result<usize> {
        let mut removed = 0;
        let threshold = std::time::Duration::from_secs(max_age_days * 24 * 60 * 60);
        let now = std::time::SystemTime::now();

        for entry in std::fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let model_id = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Skip models we want to keep
                if keep_model_ids.contains(&model_id) {
                    continue;
                }

                // Check age
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if let Ok(age) = now.duration_since(modified) {
                            if age > threshold {
                                std::fs::remove_dir_all(&path)?;
                                removed += 1;
                                info!("Removed old model cache: {}", model_id);
                            }
                        }
                    }
                }
            }
        }

        Ok(removed)
    }
}

/// Verify a model directory contains valid model files
pub fn verify_model(model_path: &Path) -> Result<ModelVerification> {
    info!("Verifying model at {:?}", model_path);

    let mut verification = ModelVerification {
        format: ModelFormat::SafeTensors, // Default, will be updated
        is_valid: false,
        files_found: Vec::new(),
        errors: Vec::new(),
    };

    if !model_path.exists() {
        verification.errors.push(format!("Model path does not exist: {:?}", model_path));
        return Ok(verification);
    }

    // Check for required files
    let tokenizer_path = model_path.join("tokenizer.json");
    if tokenizer_path.exists() {
        verification.files_found.push("tokenizer.json".to_string());
    } else {
        verification.errors.push("Missing tokenizer.json".to_string());
    }

    // Check for config.json (recommended for SafeTensors models)
    let config_path = model_path.join("config.json");
    if config_path.exists() {
        verification.files_found.push("config.json".to_string());
    }

    // Check for SafeTensors files (priority)
    let safetensors_files: Vec<_> = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    if !safetensors_files.is_empty() {
        verification.format = ModelFormat::SafeTensors;
        verification.files_found.extend(safetensors_files);
        info!("Found SafeTensors model with {} file(s)", 
            verification.files_found.len() - if config_path.exists() { 2 } else { 1 });
    }

    // Check for GGUF file (fallback)
    let gguf_files: Vec<_> = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    if !gguf_files.is_empty() {
        if verification.format != ModelFormat::SafeTensors {
            verification.format = ModelFormat::Gguf;
        }
        verification.files_found.extend(gguf_files);
    }

    // Determine validity
    verification.is_valid = !verification.files_found.is_empty() 
        && verification.errors.is_empty();

    if verification.is_valid {
        info!("Model verification passed: {} format", verification.format.as_str());
    } else {
        warn!("Model verification failed with {} errors", verification.errors.len());
    }

    Ok(verification)
}

/// Download a model from HuggingFace Hub
#[cfg(feature = "model-download")]
pub async fn download_model(
    model_id: &str,
    cache_dir: &Path,
    files: Option<&[&str]>,
    config: Option<DownloadConfig>,
) -> Result<PathBuf> {
    let config = config.unwrap_or_default();
    let model_path = cache_dir.join(model_id.replace('/', "--"));
    std::fs::create_dir_all(&model_path)?;

    // Determine which files to download
    let files_to_download = match files {
        Some(f) => f.to_vec(),
        None => vec![
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "decoder.pth",
        ],
    };

    info!("Downloading model {} to {:?}", model_id, model_path);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.timeout_secs))
        .build()?;

    for file in &files_to_download {
        let file_path = model_path.join(file);

        // Check if file already exists
        if file_path.exists() {
            info!("File {} already exists, skipping download", file);
            continue;
        }

        let url = format!("{}/{}/resolve/main/{}", config.hf_endpoint, model_id, file);

        // Try to download with retries
        let mut last_error = None;
        for attempt in 0..config.max_retries {
            match download_file_with_progress(
                &client,
                &url,
                &file_path,
                config.show_progress,
                config.chunk_size,
            )
            .await
            {
                Ok(_) => {
                    info!("Successfully downloaded {}", file);
                    break;
                }
                Err(e) => {
                    warn!("Download attempt {} failed for {}: {}", attempt + 1, file, e);
                    last_error = Some(e);
                    if attempt < config.max_retries - 1 {
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await;
                    }
                }
            }
        }

        if let Some(e) = last_error {
            return Err(e);
        }
    }

    // Verify checksums if available
    if config.verify_checksums {
        if let Err(e) = verify_model_checksums(&model_path).await {
            warn!("Checksum verification failed: {}", e);
        }
    }

    Ok(model_path)
}

/// Download file with progress bar
#[cfg(feature = "model-download")]
async fn download_file_with_progress(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
    show_progress: bool,
    _chunk_size: usize,
) -> Result<()> {
    // Create parent directories
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Send HEAD request to get file size
    let head_response = client.head(url).send().await?;
    let total_size = head_response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    // Create progress bar
    let progress_bar = if show_progress {
        let pb = ProgressBar::new(total_size.unwrap_or(0));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("Invalid progress bar template")
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Download the file
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(ModelLoaderError::Network(
            response.error_for_status().unwrap_err(),
        ));
    }

    // Create temp file
    let temp_path = dest_path.with_extension("tmp");
    let mut file = tokio::fs::File::create(&temp_path).await?;

    // Stream download with progress
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if let Some(ref pb) = progress_bar {
            pb.set_position(downloaded);
        }
    }

    file.flush().await?;
    drop(file);

    if let Some(ref pb) = progress_bar {
        pb.finish_with_message("Download complete");
    }

    // Move temp file to final destination
    tokio::fs::rename(&temp_path, dest_path).await?;

    Ok(())
}

/// Verify model checksums from sha256 file if available
#[cfg(feature = "model-download")]
async fn verify_model_checksums(model_path: &Path) -> Result<()> {
    use sha2::{Digest, Sha256};

    let checksum_file = model_path.join("sha256_checksums.txt");
    if !checksum_file.exists() {
        return Ok(());
    }

    let content = tokio::fs::read_to_string(&checksum_file).await?;
    let mut checksums: HashMap<String, String> = HashMap::new();

    for line in content.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 2 {
            checksums.insert(parts[1].to_string(), parts[0].to_string());
        }
    }

    for (filename, expected_hash) in checksums {
        let file_path = model_path.join(&filename);
        if !file_path.exists() {
            continue;
        }

        let mut file = std::fs::File::open(&file_path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let result = hasher.finalize();
        let actual_hash = hex::encode(result);

        if actual_hash != expected_hash {
            return Err(ModelLoaderError::ChecksumMismatch {
                file: filename,
                expected: expected_hash,
                actual: actual_hash,
            });
        }

        debug!("Checksum verified for {}", filename);
    }

    Ok(())
}

/// Load SafeTensors model weights
pub fn load_safetensors_model(model_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    info!("Loading SafeTensors model from {:?}", model_path);

    // Find all SafeTensors files
    let mut safetensors_files: Vec<_> = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    if safetensors_files.is_empty() {
        return Err(ModelLoaderError::NotFound(
            "No .safetensors files found in model directory".to_string()
        ));
    }

    // Sort files for consistent loading
    safetensors_files.sort();

    // Load all tensors from all files
    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();

    for file_path in safetensors_files {
        info!("Loading tensors from {:?}", file_path);
        let file = std::fs::File::open(&file_path)?;
        let buffer = unsafe { memmap2::Mmap::map(&file)? };

        let tensors = candle_core::safetensors::load_buffer(&buffer, device)
            .map_err(ModelLoaderError::Candle)?;

        for (name, tensor) in tensors {
            all_tensors.insert(name, tensor);
        }
    }

    info!("Loaded {} tensors from SafeTensors model", all_tensors.len());
    Ok(all_tensors)
}

/// Load quantized model weights from GGUF file
pub fn load_gguf_model_weights(path: &Path, device: &Device) -> Result<candle_transformers::models::quantized_llama::ModelWeights> {
    info!("Loading GGUF model from {:?}", path);

    // Handle directory path
    let gguf_path = if path.is_dir() {
        find_file_in_dir(path, "gguf")?
    } else {
        path.to_path_buf()
    };

    // Validate format
    match ModelFormat::from_path(&gguf_path)? {
        ModelFormat::Gguf => {}
        format => {
            return Err(ModelLoaderError::InvalidFormat(format!(
                "Expected GGUF format, got {:?}",
                format
            )));
        }
    }

    // Load GGUF file - first read the content, then load the model
    let file = std::fs::File::open(&gguf_path)?;
    let mut reader = std::io::BufReader::new(file);
    
    // Read GGUF content
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut reader)
        .map_err(ModelLoaderError::Candle)?;
    
    // Load model weights from GGUF content
    let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
        gguf_content,
        &mut reader,
        device,
    )?;

    info!("Successfully loaded GGUF model");
    Ok(model)
}

/// Load model weights with automatic format detection
/// 
/// Priority: SafeTensors > GGUF
pub fn load_model_weights(model_path: &Path) -> Result<ModelFormat> {
    info!("Loading model weights from {:?}", model_path);

    // Check for SafeTensors files first (priority)
    let safetensors_count = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .count();

    if safetensors_count > 0 {
        info!("Detected SafeTensors format with {} file(s)", safetensors_count);
        return Ok(ModelFormat::SafeTensors);
    }

    // Check for GGUF file (fallback)
    let gguf_count = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .count();

    if gguf_count > 0 {
        info!("Detected GGUF format");
        return Ok(ModelFormat::Gguf);
    }

    Err(ModelLoaderError::NotFound(
        "No supported model files found. Expected: *.safetensors or *.gguf".to_string()
    ))
}

/// Load decoder weights from file
pub fn load_decoder_weights(path: &Path, device: &Device) -> Result<Tensor> {
    info!("Loading decoder weights from {:?}", path);

    // Handle directory path
    let weights_path = if path.is_dir() {
        find_file_in_dir(path, "safetensors")
            .or_else(|_| find_file_in_dir(path, "pth"))
            .or_else(|_| find_file_in_dir(path, "pt"))?
    } else {
        path.to_path_buf()
    };

    let format = ModelFormat::from_path(&weights_path)?;

    match format {
        ModelFormat::SafeTensors => load_safetensors_single(&weights_path, device),
        ModelFormat::PyTorch => {
            // Convert PyTorch format to Candle tensors
            warn!("PyTorch format detected, attempting conversion");
            load_pytorch_weights(&weights_path, device)
        }
        _ => Err(ModelLoaderError::InvalidFormat(format!(
            "Unsupported decoder format: {:?}",
            format
        ))),
    }
}

/// Load a single tensor from SafeTensors file
fn load_safetensors_single(path: &Path, device: &Device) -> Result<Tensor> {
    let file = std::fs::File::open(path)?;
    let buffer = unsafe { memmap2::Mmap::map(&file)? };

    let tensors = candle_core::safetensors::load_buffer(&buffer, device)
        .map_err(ModelLoaderError::Candle)?;

    // Return the first tensor as the decoder weights
    // In practice, you'd want to identify the specific tensor name
    tensors
        .values()
        .next()
        .cloned()
        .ok_or_else(|| ModelLoaderError::NotFound("No tensors found in safetensors file".to_string()))
}

/// Load PyTorch weights and convert to Candle format
fn load_pytorch_weights(path: &Path, device: &Device) -> Result<Tensor> {
    // PyTorch pickle format is complex - for production, use PyTorchFile
    // This is a simplified version that attempts basic loading
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    // PyTorch files start with a specific magic number
    let mut lines = reader.lines();
    if let Some(Ok(first_line)) = lines.next() {
        if first_line.contains("PK") {
            // It's a ZIP file (PyTorch's default format)
            return load_pytorch_zip(path, device);
        }
    }

    Err(ModelLoaderError::InvalidFormat(
        "Unable to parse PyTorch file format".to_string(),
    ))
}

/// Load PyTorch weights from ZIP format
fn load_pytorch_zip(path: &Path, _device: &Device) -> Result<Tensor> {
    #[cfg(feature = "model-download")]
    {
        use zip::ZipArchive;

        let file = std::fs::File::open(path)?;
        let mut archive = ZipArchive::new(file)?;

        // Look for the data.pkl file which contains the tensor data
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name();

            if name == "data.pkl" || name.ends_with(".pkl") {
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer)?;

                // For now, return an error indicating conversion is needed
                // Full PyTorch pickle parsing requires additional dependencies
                return Err(ModelLoaderError::InvalidFormat(
                    "PyTorch format requires conversion. Use convert_pytorch_to_safetensors()".to_string(),
                ));
            }
        }
    }

    Err(ModelLoaderError::InvalidFormat(
        "Could not find tensor data in PyTorch file".to_string(),
    ))
}

/// Find a file with specific extension in directory
fn find_file_in_dir(dir: &Path, extension: &str) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == extension {
                    return Ok(path);
                }
            }
        }
    }

    Err(ModelLoaderError::NotFound(format!(
        "No .{} file found in {:?}",
        extension, dir
    )))
}

/// Convert PyTorch .pth file to SafeTensors format
/// 
/// This requires the `model-download` feature and additional dependencies
/// for full pickle parsing. For production use, consider using:
/// - The `pytorch-file` crate
/// - External conversion tools like `convert-to-safetensors`
#[cfg(feature = "model-download")]
pub fn convert_pytorch_to_safetensors(
    pytorch_path: &Path,
    output_path: &Path,
) -> Result<()> {
    info!(
        "Converting {:?} to SafeTensors format at {:?}",
        pytorch_path, output_path
    );

    // This is a placeholder for the conversion logic
    // In production, use a proper PyTorch pickle parser
    // or call an external conversion tool

    Err(ModelLoaderError::Other(
        "PyTorch to SafeTensors conversion requires the 'pytorch-convert' feature. \
         Please install the model with: pip install safetensors torch".to_string(),
    ))
}

/// Load a complete model from HuggingFace Hub or local cache
///
/// This is the main entry point for model loading in production.
#[cfg(feature = "model-download")]
pub async fn load_model_from_hf(
    model_id: &str,
    cache_dir: Option<&Path>,
) -> Result<ModelFormat> {
    let cache_dir = match cache_dir {
        Some(path) => path.to_path_buf(),
        None => ModelCache::default_cache_dir()?,
    };

    // Download or use cached model
    let model_path = download_model(
        model_id,
        &cache_dir,
        None,
        None,
    )
    .await?;

    // Verify model
    let verification = verify_model(&model_path)?;
    if !verification.is_valid {
        return Err(ModelLoaderError::InvalidFormat(
            format!("Model verification failed: {:?}", verification.errors)
        ));
    }

    Ok(verification.format)
}

/// Load model from a local path with automatic format detection
pub fn load_model_from_path(
    path: &Path,
) -> Result<ModelFormat> {
    info!("Loading model from local path: {:?}", path);

    if !path.exists() {
        return Err(ModelLoaderError::NotFound(format!(
            "Model path does not exist: {:?}",
            path
        )));
    }

    // Verify model
    let verification = verify_model(path)?;
    if !verification.is_valid {
        return Err(ModelLoaderError::InvalidFormat(
            format!("Model verification failed: {:?}", verification.errors)
        ));
    }

    Ok(verification.format)
}

/// Information about available models
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub description: String,
    pub size_mb: u64,
    pub tags: Vec<String>,
}

/// List available pretrained models
pub fn list_available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            model_id: "ekwek/Soprano-1.1-80M".to_string(),
            description: "Soprano TTS model (80M params, SafeTensors format)".to_string(),
            size_mb: 320,
            tags: vec!["base".to_string(), "safetensors".to_string()],
        },
        ModelInfo {
            model_id: "hexgrad/soprano-base".to_string(),
            description: "Base Soprano TTS model (1B params, 4-bit quantized)".to_string(),
            size_mb: 2048,
            tags: vec!["base".to_string(), "quantized".to_string()],
        },
        ModelInfo {
            model_id: "hexgrad/soprano-small".to_string(),
            description: "Small Soprano TTS model (400M params, 4-bit quantized)".to_string(),
            size_mb: 1024,
            tags: vec!["small".to_string(), "fast".to_string()],
        },
    ]
}

/// Utility to check if CUDA is available
pub fn cuda_available() -> bool {
    candle_core::utils::cuda_is_available()
}

/// Get the best available device
pub fn get_optimal_device() -> Device {
    if cuda_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

// Re-export Tokenizer for convenience
pub use tokenizers::Tokenizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_format_detection() {
        let path = PathBuf::from("model.gguf");
        assert_eq!(ModelFormat::from_path(&path).unwrap(), ModelFormat::Gguf);

        let path = PathBuf::from("model.safetensors");
        assert_eq!(
            ModelFormat::from_path(&path).unwrap(),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_model_format_as_str() {
        assert_eq!(ModelFormat::SafeTensors.as_str(), "SafeTensors");
        assert_eq!(ModelFormat::Gguf.as_str(), "GGUF");
        assert_eq!(ModelFormat::PyTorch.as_str(), "PyTorch");
    }

    #[test]
    fn test_model_cache() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(temp_dir.path()).unwrap();

        let model_path = cache.model_path("test/model");
        assert!(model_path.to_str().unwrap().contains("test--model"));
    }

    #[test]
    fn test_list_available_models() {
        let models = list_available_models();
        assert!(!models.is_empty());
        
        // Check for SafeTensors model
        let has_safetensors = models.iter().any(|m| m.tags.contains(&"safetensors".to_string()));
        assert!(has_safetensors, "Should include SafeTensors model");
    }

    #[test]
    fn test_verify_model_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let verification = verify_model(temp_dir.path()).unwrap();
        assert!(!verification.is_valid);
        assert!(!verification.errors.is_empty());
    }
}

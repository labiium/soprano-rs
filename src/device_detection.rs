//! Automatic device detection and selection for optimal performance
//!
//! This module provides intelligent device selection based on platform and available hardware.
//! Priority order:
//! - macOS: Metal > CPU
//! - Linux: CUDA > CPU
//! - Windows: CUDA > CPU
//!
//! MLX is not supported (Metal is the only Apple GPU backend today).

use candle_core::Device;
use tracing::{debug, info, warn};

/// Information about available compute devices
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: String,
    pub platform: Platform,
    pub available: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cuda,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    MacOS,
    Linux,
    Windows,
    Other,
}

impl DeviceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::Cuda => "CUDA",
            DeviceType::Metal => "Metal",
            DeviceType::Cpu => "CPU",
        }
    }
}

impl Platform {
    pub fn current() -> Self {
        #[cfg(target_os = "macos")]
        return Platform::MacOS;
        #[cfg(target_os = "linux")]
        return Platform::Linux;
        #[cfg(target_os = "windows")]
        return Platform::Windows;
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        return Platform::Other;
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Platform::MacOS => "macOS",
            Platform::Linux => "Linux",
            Platform::Windows => "Windows",
            Platform::Other => "Other",
        }
    }
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    candle_core::utils::cuda_is_available()
}

/// Check if Metal is available
pub fn is_metal_available() -> bool {
    candle_core::utils::metal_is_available()
}

/// Get detailed information about all available devices
pub fn get_all_device_info() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();
    let platform = Platform::current();

    // Check CUDA
    let cuda_available = is_cuda_available();
    devices.push(DeviceInfo {
        device_type: DeviceType::Cuda,
        name: "NVIDIA CUDA".to_string(),
        platform,
        available: cuda_available,
        error_message: if cuda_available {
            None
        } else {
            Some("CUDA not available on this system".to_string())
        },
    });

    // Check Metal (only relevant on macOS)
    let metal_available = is_metal_available();
    devices.push(DeviceInfo {
        device_type: DeviceType::Metal,
        name: "Apple Metal".to_string(),
        platform,
        available: metal_available,
        error_message: if metal_available {
            None
        } else {
            Some("Metal not available on this system".to_string())
        },
    });

    // CPU is always available
    devices.push(DeviceInfo {
        device_type: DeviceType::Cpu,
        name: "CPU".to_string(),
        platform,
        available: true,
        error_message: None,
    });

    devices
}

/// Automatically select the best available device for the current platform
///
/// Priority order:
/// - macOS: Metal > CPU
/// - Linux: CUDA > CPU  
/// - Windows: CUDA > CPU
pub fn auto_select_device() -> Device {
    let platform = Platform::current();

    info!(platform = %platform.as_str(), "Auto-selecting best compute device");

    match platform {
        Platform::MacOS => {
            debug!("macOS detected, checking for Metal support...");
            if is_metal_available() {
                match Device::new_metal(0) {
                    Ok(device) => {
                        info!("Successfully selected Metal (Apple Silicon/AMD GPU)");
                        return device;
                    }
                    Err(e) => {
                        warn!(error = %e, "Metal reported available but failed to initialize, falling back to CPU");
                    }
                }
            } else {
                debug!("Metal not available on this macOS system");
            }
        }
        Platform::Linux | Platform::Windows => {
            debug!(
                "{} detected, checking for CUDA support...",
                platform.as_str()
            );
            if is_cuda_available() {
                match Device::new_cuda(0) {
                    Ok(device) => {
                        info!("Successfully selected CUDA (NVIDIA GPU)");
                        return device;
                    }
                    Err(e) => {
                        warn!(error = %e, "CUDA reported available but failed to initialize, falling back to CPU");
                    }
                }
            } else {
                debug!("CUDA not available on this system");
            }
        }
        Platform::Other => {
            debug!("Unknown platform, checking all acceleration options...");

            // Try Metal first (in case it's a BSD or other Unix with Metal)
            if is_metal_available() {
                if let Ok(device) = Device::new_metal(0) {
                    info!("Successfully selected Metal");
                    return device;
                }
            }

            // Try CUDA
            if is_cuda_available() {
                if let Ok(device) = Device::new_cuda(0) {
                    info!("Successfully selected CUDA");
                    return device;
                }
            }
        }
    }

    // Fallback to CPU
    info!("No GPU acceleration available, using CPU");
    Device::Cpu
}

/// Get the recommended device type for the current platform
pub fn get_recommended_device_type() -> DeviceType {
    let platform = Platform::current();

    match platform {
        Platform::MacOS => {
            if is_metal_available() {
                DeviceType::Metal
            } else {
                DeviceType::Cpu
            }
        }
        Platform::Linux | Platform::Windows => {
            if is_cuda_available() {
                DeviceType::Cuda
            } else {
                DeviceType::Cpu
            }
        }
        Platform::Other => {
            if is_metal_available() {
                DeviceType::Metal
            } else if is_cuda_available() {
                DeviceType::Cuda
            } else {
                DeviceType::Cpu
            }
        }
    }
}

/// Parse device string with support for "auto" selection
///
/// Supported values:
/// - "auto" - Automatically select best device (recommended)
/// - "cuda" or "gpu" - Use NVIDIA CUDA
/// - "metal" or "mps" - Use Apple Metal
/// - "cpu" - Use CPU only
pub fn parse_device_auto(device_str: &str) -> Device {
    let device_lower = device_str.to_lowercase();

    if device_lower == "auto" || device_lower == "best" {
        return auto_select_device();
    }

    // Manual device selection
    match device_lower.as_str() {
        "cuda" | "gpu" | "nvidia" => {
            debug!("Attempting to use CUDA device...");
            match Device::new_cuda(0) {
                Ok(device) => {
                    info!("Successfully initialized CUDA device");
                    device
                }
                Err(e) => {
                    warn!(error = %e, "CUDA initialization failed, falling back to auto-selection");
                    auto_select_device()
                }
            }
        }
        "metal" | "mps" | "apple" => {
            debug!("Attempting to use Metal device...");
            match Device::new_metal(0) {
                Ok(device) => {
                    info!("Successfully initialized Metal device");
                    device
                }
                Err(e) => {
                    warn!(error = %e, "Metal initialization failed, falling back to auto-selection");
                    auto_select_device()
                }
            }
        }
        "cpu" => {
            info!("Using CPU device (as requested)");
            Device::Cpu
        }
        _ => {
            warn!(device = %device_str, "Unknown device type, using auto-selection");
            auto_select_device()
        }
    }
}

/// Print a summary of available devices
pub fn print_device_summary() {
    let platform = Platform::current();
    let devices = get_all_device_info();

    println!("╔═══════════════════════════════════════════════════╗");
    println!("║         Compute Device Information                ║");
    println!("╠═══════════════════════════════════════════════════╣");
    println!("║ Platform: {:<41} ║", platform.as_str());
    println!("╠═══════════════════════════════════════════════════╣");

    for device in &devices {
        let status = if device.available { "✓" } else { "✗" };
        let device_name = format!("{:<20}", device.name);
        println!("║ {} {} ║", status, device_name);
    }

    println!("╠═══════════════════════════════════════════════════╣");

    let recommended = get_recommended_device_type();
    println!("║ Recommended: {:<36} ║", recommended.as_str());

    let selected = auto_select_device();
    let selected_name = match selected {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal",
    };
    println!("║ Auto-selected: {:<34} ║", selected_name);

    println!("╚═══════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = Platform::current();
        // Just verify it doesn't panic
        assert!(!platform.as_str().is_empty());
    }

    #[test]
    fn test_device_type_as_str() {
        assert_eq!(DeviceType::Cuda.as_str(), "CUDA");
        assert_eq!(DeviceType::Metal.as_str(), "Metal");
        assert_eq!(DeviceType::Cpu.as_str(), "CPU");
    }

    #[test]
    fn test_get_all_device_info() {
        let devices = get_all_device_info();
        assert!(!devices.is_empty());

        // Should have at least CPU
        let has_cpu = devices.iter().any(|d| d.device_type == DeviceType::Cpu);
        assert!(has_cpu, "CPU should always be reported");
    }

    #[test]
    fn test_auto_select_device() {
        let device = auto_select_device();
        // Just verify it returns a valid device
        // The actual device type depends on the hardware
        assert!(matches!(
            device,
            Device::Cpu | Device::Cuda(_) | Device::Metal(_)
        ));
    }

    #[test]
    fn test_parse_device_auto() {
        // Test auto mode
        let device = parse_device_auto("auto");
        assert!(matches!(
            device,
            Device::Cpu | Device::Cuda(_) | Device::Metal(_)
        ));

        // Test explicit CPU
        let device = parse_device_auto("cpu");
        assert!(matches!(device, Device::Cpu));

        // Test case insensitivity
        let device = parse_device_auto("CPU");
        assert!(matches!(device, Device::Cpu));

        let device = parse_device_auto("Auto");
        assert!(matches!(
            device,
            Device::Cpu | Device::Cuda(_) | Device::Metal(_)
        ));
    }

    #[test]
    fn test_cuda_availability_check() {
        // Just verify it doesn't panic
        let _ = is_cuda_available();
    }

    #[test]
    fn test_metal_availability_check() {
        // Just verify it doesn't panic
        let _ = is_metal_available();
    }
}

//! CoreX device information and utility functions

use super::{CorexError, CorexFeature};
use crate::{DType, Result};
use std::fmt;

/// CoreX device capabilities and properties
#[derive(Debug, Clone)]
pub struct CorexDeviceInfo {
    /// GPU device ID
    pub gpu_id: usize,
    /// Compute capability (major, minor)
    pub compute_capability: (u8, u8),
    /// Total memory size in bytes
    pub memory_size: usize,
    /// CoreX SDK version
    pub corex_version: String,
    /// Supports BF16 operations
    pub supports_bf16: bool,
    /// Supports FP8 operations
    pub supports_fp8: bool,
    /// Supports Tensor Core operations
    pub supports_tensor_cores: bool,
    /// Maximum threads per block
    pub max_threads_per_block: usize,
    /// Maximum grid dimensions
    pub max_grid_dim: [u32; 3],
    /// Maximum block dimensions
    pub max_block_dim: [u32; 3],
    /// Warp size
    pub warp_size: usize,
}

impl CorexDeviceInfo {
    /// Enhance device info with CoreX-specific information
    pub fn enhance_with_corex_info(&mut self) -> Result<()> {
        // This would query CoreX SDK for specific information
        // For now, we'll add some reasonable defaults
        
        // Check for specific CoreX optimizations
        if self.corex_version.starts_with("4.") {
            // CoreX 4.x series has specific optimizations
        }
        
        Ok(())
    }
    
    /// Check if device supports specific precision
    pub fn supports_precision(&self, dtype: DType) -> bool {
        match dtype {
            DType::F16 | DType::BF16 => self.supports_bf16,
            DType::F8E4M3 | DType::F8E5M2 => self.supports_fp8,
            _ => true, // F32 and others are always supported
        }
    }
    
    /// Get optimal block size for given operation
    pub fn optimal_block_size(&self, operation_type: OperationType) -> usize {
        match operation_type {
            OperationType::MatMul => {
                // Optimize for matmul based on compute capability
                if self.compute_capability >= (8, 0) {
                    256 // Ampere and later
                } else if self.compute_capability >= (7, 0) {
                    128 // Turing
                } else {
                    64 // Pascal and earlier
                }
            }
            OperationType::Convolution => {
                if self.supports_tensor_cores {
                    128
                } else {
                    64
                }
            }
            OperationType::ElementWise => {
                self.warp_size
            }
            OperationType::Reduction => {
                256
            }
        }
    }
    
    /// Get memory bandwidth estimate in GB/s
    pub fn estimated_bandwidth(&self) -> f64 {
        // This would be calculated from CoreX specifications
        match self.compute_capability {
            (8, _) => 1550.0, // Ampere
            (7, _) => 900.0,  // Turing
            _ => 500.0,       // Older architectures
        }
    }
}

impl fmt::Display for CorexDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CoreX Device Information:")?;
        writeln!(f, "  GPU ID: {}", self.gpu_id)?;
        writeln!(f, "  Compute Capability: {}.{}", 
                 self.compute_capability.0, self.compute_capability.1)?;
        writeln!(f, "  Memory Size: {} MB", self.memory_size / 1024 / 1024)?;
        writeln!(f, "  CoreX Version: {}", self.corex_version)?;
        writeln!(f, "  BF16 Support: {}", self.supports_bf16)?;
        writeln!(f, "  FP8 Support: {}", self.supports_fp8)?;
        writeln!(f, "  Tensor Cores: {}", self.supports_tensor_cores)?;
        writeln!(f, "  Max Threads per Block: {}", self.max_threads_per_block)?;
        writeln!(f, "  Warp Size: {}", self.warp_size)
    }
}

/// CoreX-specific features that can be enabled/disabled
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorexFeature {
    /// Brain Float 16 precision
    BF16,
    /// 8-bit Floating Point precision
    FP8,
    /// Tensor Core acceleration
    TensorCores,
    /// Memory pooling
    MemoryPool,
    /// Asynchronous operations
    AsyncOperations,
    /// Kernel caching
    KernelCache,
    /// Custom CUDA kernels
    CustomKernels,
}

impl fmt::Display for CorexFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorexFeature::BF16 => write!(f, "BF16"),
            CorexFeature::FP8 => write!(f, "FP8"),
            CorexFeature::TensorCores => write!(f, "Tensor Cores"),
            CorexFeature::MemoryPool => write!(f, "Memory Pool"),
            CorexFeature::AsyncOperations => write!(f, "Async Operations"),
            CorexFeature::KernelCache => write!(f, "Kernel Cache"),
            CorexFeature::CustomKernels => write!(f, "Custom Kernels"),
        }
    }
}

/// Types of GPU operations
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    /// Matrix multiplication
    MatMul,
    /// Convolution operations
    Convolution,
    /// Element-wise operations
    ElementWise,
    /// Reduction operations
    Reduction,
}

/// Detect all available CoreX devices
pub fn detect_corex_devices() -> Result<Vec<CorexDeviceInfo>> {
    let mut devices = Vec::new();
    
    if !is_corex_available() {
        return Ok(devices);
    }
    
    // Get device count
    let device_count = cudarc::driver::device_count().unwrap_or(0);
    
    for gpu_id in 0..device_count {
        // Check if this is a CoreX device
        if is_corex_device(gpu_id)? {
            let device_info = query_device_info(gpu_id)?;
            devices.push(device_info);
        }
    }
    
    Ok(devices)
}

/// Check if a specific GPU ID corresponds to a CoreX device
pub fn is_corex_device(gpu_id: usize) -> Result<bool> {
    // This would use CoreX SDK to detect if a device is CoreX-compatible
    // For now, we'll assume all CUDA devices could be CoreX devices
    // if the environment is properly configured
    
    if !is_corex_available() {
        return Ok(false);
    }
    
    // Try to get device properties
    match cudarc::driver::get_device_props(gpu_id) {
        Ok(props) => {
            // Check device name for CoreX indicators
            let name = props.name.to_str().unwrap_or("");
            Ok(name.contains("Iluvatar") || 
               name.contains("CoreX") || 
               is_corex_environment_configured())
        }
        Err(_) => Ok(false),
    }
}

/// Check if CoreX environment is properly configured
fn is_corex_environment_configured() -> bool {
    std::env::var("COREX_HOME").is_ok() ||
    std::env::var("COREX_ROOT").is_ok() ||
    std::path::Path::new("/opt/iluvatar/corex").exists()
}

/// Query device information for a specific GPU
pub fn query_device_info(gpu_id: usize) -> Result<CorexDeviceInfo> {
    let device_props = cudarc::driver::get_device_props(gpu_id)?;
    
    let compute_cap = (
        ((device_props.major) as u8),
        ((device_props.minor) as u8),
    );
    
    let mut device_info = CorexDeviceInfo {
        gpu_id,
        compute_capability: compute_cap,
        memory_size: device_props.total_memory,
        corex_version: get_corex_version()?,
        supports_bf16: compute_cap >= (7, 0),
        supports_fp8: compute_cap >= (8, 9),
        supports_tensor_cores: compute_cap >= (7, 0),
        max_threads_per_block: device_props.max_threads_per_block as usize,
        max_grid_dim: device_props.max_grid_dim,
        max_block_dim: device_props.max_block_dim,
        warp_size: device_props.warp_size as usize,
    };
    
    device_info.enhance_with_corex_info()?;
    Ok(device_info)
}

/// Get CoreX SDK version
pub fn get_corex_version() -> Result<String> {
    // This would query CoreX SDK for version information
    // For now, check environment variables or use default
    
    if let Ok(version) = std::env::var("COREX_VERSION") {
        Ok(version)
    } else if let Ok(corex_home) = std::env::var("COREX_HOME") {
        // Could read from a version file in the SDK
        Ok("4.3.8".to_string())
    } else {
        Ok("Unknown".to_string())
    }
}

/// Check if CoreX backend is available
pub fn is_corex_available() -> bool {
    // Check multiple indicators for CoreX availability
    
    // 1. Check for CoreX SDK installation
    if let Ok(corex_path) = std::env::var("COREX_HOME") {
        let lib_path = std::path::Path::new(&corex_path).join("lib64");
        if !lib_path.exists() {
            return false;
        }
        
        // Check for key libraries
        let required_libs = ["libcuda.so", "libcublas.so", "libcudart.so"];
        for lib in required_libs {
            if !lib_path.join(lib).exists() {
                return false;
            }
        }
        return true;
    }
    
    // 2. Check for system-wide installation
    let system_paths = [
        "/opt/iluvatar/corex",
        "/usr/local/corex",
        "/opt/corex",
    ];
    
    for path in system_paths.iter() {
        let lib_path = std::path::Path::new(path).join("lib64");
        if lib_path.exists() && lib_path.join("libcuda.so").exists() {
            return true;
        }
    }
    
    // 3. Check if CUDA runtime is available (minimum requirement)
    cudarc::driver::device_count().is_ok()
}

/// Get recommended configuration for the device
pub fn get_recommended_config(device_info: &CorexDeviceInfo) -> DeviceConfig {
    DeviceConfig {
        gpu_id: device_info.gpu_id,
        enable_memory_pool: device_info.memory_size > 4 * 1024 * 1024 * 1024, // > 4GB
        memory_pool_size: Some((device_info.memory_size as f64 * 0.8) as usize), // 80% of total
        preferred_precision: if device_info.supports_fp8 {
            Some(DType::F8E5M2)
        } else if device_info.supports_bf16 {
            Some(DType::BF16)
        } else {
            None
        },
        enable_async: true,
        stream_priority: 0,
    }
}

/// Recommended device configuration
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub gpu_id: usize,
    pub enable_memory_pool: bool,
    pub memory_pool_size: Option<usize>,
    pub preferred_precision: Option<DType>,
    pub enable_async: bool,
    pub stream_priority: i32,
}

/// Validate device configuration
pub fn validate_device_config(config: &DeviceConfig) -> Result<()> {
    // Validate GPU ID
    let device_count = cudarc::driver::device_count().unwrap_or(0);
    if config.gpu_id >= device_count {
        return Err(CorexError::DeviceNotFound { 
            gpu_id: config.gpu_id 
        }.into());
    }
    
    // Validate memory pool size
    if let Some(pool_size) = config.memory_pool_size {
        if pool_size > 128 * 1024 * 1024 * 1024 { // 128GB max
            return Err(CorexError::Custom(
                "Memory pool size too large (max 128GB)".to_string()
            ).into());
        }
    }
    
    Ok(())
}
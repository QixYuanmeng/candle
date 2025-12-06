//! CoreX GPU backend for Candle framework
//!
//! This module provides CoreX GPU acceleration support by extending the
//! CUDA backend with CoreX-specific optimizations and device management.
//!
//! # Features
//!
//! - CoreX device detection and initialization
//! - CUDA 10.2 compatibility layer
//! - CoreX-specific memory management optimizations
//! - Hardware-accelerated tensor operations
//! - BF16/FP8 precision support
//!
//! # Example
//!
//! ```rust
//! use candle_core::{Tensor, Device};
//!
//! // Create a CoreX device
//! let device = Device::new_corex(0)?;
//!
//! // Perform GPU-accelerated operations
//! let a = Tensor::arange(0f32, 1000f32, &device)?;
//! let b = Tensor::arange(0f32, 1000f32, &device)?;
//! let c = a.matmul(&b.reshape((1000, 1))?)?;
//! ```

use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, Storage, WithDType};
use std::sync::Arc;
use std::collections::HashMap;

// Re-export main types for convenience
pub use device::CorexDevice;
pub use storage::CorexStorage;
pub use utils::{CorexDeviceInfo, CorexFeature, detect_corex_devices};

mod device;
mod storage;
mod utils;
#[cfg(test)]
mod tests;

/// CoreX backend error types
#[derive(Debug, thiserror::Error)]
pub enum CorexError {
    #[error("CoreX device not found: {gpu_id}")]
    DeviceNotFound { gpu_id: usize },
    
    #[error("CoreX SDK not found or incompatible")]
    SdkNotFound,
    
    #[error("CoreX feature not supported: {feature:?}")]
    UnsupportedFeature { feature: CorexFeature },
    
    #[error("CoreX memory allocation failed: {size} bytes")]
    MemoryAllocationFailed { size: usize },
    
    #[error("CoreX CUDA error: {0}")]
    CudaError(#[from] cudarc::driver::CudaError),
    
    #[error("CoreX specific error: {0}")]
    Custom(String),
}

impl From<CorexError> for Error {
    fn from(err: CorexError) -> Self {
        Error::wrap(err)
    }
}

/// CoreX backend configuration
#[derive(Debug, Clone)]
pub struct CorexConfig {
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    
    /// Enable memory optimizations
    pub enable_memory_optimization: bool,
    
    /// Enable kernel caching
    pub enable_kernel_cache: bool,
    
    /// Preferred precision
    pub preferred_precision: Option<DType>,
    
    /// Additional environment variables
    pub env_vars: HashMap<String, String>,
}

impl Default for CorexConfig {
    fn default() -> Self {
        let mut env_vars = HashMap::new();
        
        // CoreX performance optimizations
        env_vars.insert("COREX_MEMORY_POOL".to_string(), "1".to_string());
        env_vars.insert("COREX_KERNEL_CACHE".to_string(), "1".to_string());
        env_vars.insert("COREX_PRECISION".to_string(), "auto".to_string());
        
        Self {
            memory_pool_size: None,
            enable_memory_optimization: true,
            enable_kernel_cache: true,
            preferred_precision: None,
            env_vars,
        }
    }
}

/// Global CoreX backend state
static COREX_INITIALIZED: std::sync::Once = std::sync::Once::new();

/// Initialize CoreX backend
pub fn initialize_corex(config: CorexConfig) -> Result<()> {
    COREX_INITIALIZED.call_once(|| {
        // Set environment variables for CoreX
        if let Ok(corex_path) = std::env::var("COREX_HOME") {
            let lib_path = format!("{}/lib64", corex_path);
            let current_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            std::env::set_var("LD_LIBRARY_PATH", format!("{}:{}", lib_path, current_path));
        }
        
        // Apply configuration
        for (key, value) in config.env_vars {
            std::env::set_var(key, value);
        }
        
        // Initialize CoreX runtime (if needed)
        // This would call CoreX SDK initialization functions
    });
    
    Ok(())
}

/// Check if CoreX backend is available
pub fn is_corex_available() -> bool {
    // Check if CoreX SDK is properly installed
    if let Ok(corex_path) = std::env::var("COREX_HOME") {
        let lib_path = std::path::Path::new(&corex_path).join("lib64");
        lib_path.exists() && lib_path.join("libcuda.so").exists()
    } else {
        false
    }
}

/// Get number of available CoreX devices
pub fn corex_device_count() -> Result<usize> {
    if !is_corex_available() {
        return Ok(0);
    }
    
    // Use CUDA API to get device count, filtered for CoreX devices
    // This is a simplified implementation
    match cudarc::driver::device_count() {
        Ok(count) => Ok(count),
        Err(_) => Ok(0),
    }
}

/// Create a CoreX device if available, fallback to CUDA
pub fn create_device_with_fallback(gpu_id: usize) -> Result<crate::Device> {
    if is_corex_available() {
        match CorexDevice::new(gpu_id) {
            Ok(device) => Ok(crate::Device::Corex(device)),
            Err(_) => {
                // Fallback to CUDA if CoreX initialization fails
                Ok(crate::Device::Cuda(crate::CudaDevice::new(gpu_id)?))
            }
        }
    } else {
        Ok(crate::Device::Cuda(crate::CudaDevice::new(gpu_id)?))
    }
}
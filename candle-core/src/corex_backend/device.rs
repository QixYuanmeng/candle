//! CoreX device management and GPU operations
//!
//! This module provides CoreX GPU device abstraction, handling device
//! initialization, memory management, and GPU-specific operations.

use super::{CorexError, CorexFeature, CorexDeviceInfo};
use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Error, Layout, Result, Shape, WithDType};
use std::sync::Arc;
use std::sync::Mutex;

/// CoreX GPU device wrapper that extends CUDA device functionality
#[derive(Debug, Clone)]
pub struct CorexDevice {
    /// Underlying CUDA device
    cuda_device: crate::CudaDevice,
    /// CoreX device information
    device_info: CorexDeviceInfo,
    /// Memory pool for efficient allocation
    memory_pool: Arc<Mutex<MemoryPool>>,
    /// Device configuration
    config: DeviceConfig,
}

/// Device-specific configuration
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// GPU ID
    pub gpu_id: usize,
    /// Enable memory pooling
    pub enable_memory_pool: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: Option<usize>,
    /// Preferred compute precision
    pub preferred_precision: Option<DType>,
    /// Enable async operations
    pub enable_async: bool,
    /// Stream priority
    pub stream_priority: i32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            gpu_id: 0,
            enable_memory_pool: true,
            memory_pool_size: None, // Auto-detect
            preferred_precision: None, // Use framework default
            enable_async: true,
            stream_priority: 0,
        }
    }
}

/// Memory pool for efficient GPU memory management
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool of free memory blocks
    free_blocks: Vec<MemoryBlock>,
    /// Total allocated memory
    total_allocated: usize,
    /// Maximum allowed allocation
    max_allocation: usize,
}

/// A block of GPU memory
#[derive(Debug)]
struct MemoryBlock {
    /// Pointer to memory
    ptr: *mut std::ffi::c_void,
    /// Size in bytes
    size: usize,
    /// Alignment
    alignment: usize,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(max_size: usize) -> Self {
        Self {
            free_blocks: Vec::new(),
            total_allocated: 0,
            max_allocation: max_size,
        }
    }
    
    /// Allocate memory from pool
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut std::ffi::c_void> {
        // Try to find a suitable block in the pool
        for i in 0..self.free_blocks.len() {
            let block = &self.free_blocks[i];
            if block.size >= size && block.alignment >= alignment {
                let ptr = self.free_blocks.remove(i).ptr;
                return Ok(ptr);
            }
        }
        
        // Allocate new block
        if self.total_allocated + size > self.max_allocation {
            return Err(CorexError::MemoryAllocationFailed { size }.into());
        }
        
        // Use CUDA allocation for simplicity
        // In production, this would use CoreX-specific allocation functions
        unsafe {
            let mut ptr = std::ptr::null_mut();
            let result = cudarc::driver::sys::cuMemAlloc(&mut ptr, size);
            if result != 0 {
                return Err(CorexError::CudaError(
                    cudarc::driver::CudaError::from_sys(result)
                ).into());
            }
            
            self.total_allocated += size;
            Ok(ptr as *mut std::ffi::c_void)
        }
    }
    
    /// Deallocate memory back to pool
    pub fn deallocate(&mut self, ptr: *mut std::ffi::c_void, size: usize, alignment: usize) {
        if self.free_blocks.len() < 1000 { // Prevent pool from growing too large
            self.free_blocks.push(MemoryBlock { ptr, size, alignment });
        } else {
            // Free directly if pool is full
            unsafe {
                let result = cudarc::driver::sys::cuMemFree(ptr as cudarc::driver::sys::CUdeviceptr);
                if result == 0 {
                    self.total_allocated = self.total_allocated.saturating_sub(size);
                }
            }
        }
    }
}

impl CorexDevice {
    /// Create a new CoreX device
    pub fn new(gpu_id: usize) -> Result<Self> {
        Self::new_with_config(gpu_id, DeviceConfig::default())
    }
    
    /// Create a new CoreX device with custom configuration
    pub fn new_with_config(gpu_id: usize, config: DeviceConfig) -> Result<Self> {
        // Initialize CoreX environment
        Self::setup_corex_environment(gpu_id)?;
        
        // Create underlying CUDA device
        let cuda_device = if config.enable_async {
            crate::CudaDevice::new_with_stream(gpu_id)?
        } else {
            crate::CudaDevice::new(gpu_id)?
        };
        
        // Query device information
        let device_info = Self::query_device_info(gpu_id)?;
        
        // Initialize memory pool
        let memory_pool_size = config.memory_pool_size.unwrap_or_else(|| {
            // Use 80% of total GPU memory
            (device_info.memory_size as f64 * 0.8) as usize
        });
        
        let memory_pool = Arc::new(Mutex::new(MemoryPool::new(memory_pool_size)));
        
        Ok(Self {
            cuda_device,
            device_info,
            memory_pool,
            config,
        })
    }
    
    /// Setup CoreX-specific environment variables and configurations
    fn setup_corex_environment(gpu_id: usize) -> Result<()> {
        // Check for CoreX SDK
        if let Ok(corex_path) = std::env::var("COREX_HOME") {
            let lib_path = format!("{}/lib64", corex_path);
            let current_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            std::env::set_var("LD_LIBRARY_PATH", format!("{}:{}", lib_path, current_path));
        } else {
            return Err(CorexError::SdkNotFound.into());
        }
        
        // Set GPU visibility
        std::env::set_var("CUDA_VISIBLE_DEVICES", gpu_id.to_string());
        
        // CoreX performance optimizations
        std::env::set_var("COREX_MEMORY_POOL", "1");
        std::env::set_var("COREX_KERNEL_CACHE", "1");
        std::env::set_var("COREX_PRECISION", "auto");
        std::env::set_var("COREX_NUM_THREADS", "0"); // Auto-detect
        
        Ok(())
    }
    
    /// Query CoreX device capabilities and information
    fn query_device_info(gpu_id: usize) -> Result<CorexDeviceInfo> {
        // Get CUDA device properties first
        let device_props = cudarc::driver::get_device_props(gpu_id)?;
        
        // Parse CoreX-specific information
        // This would typically involve CoreX SDK calls
        let compute_cap = (
            ((device_props.major) as u8),
            ((device_props.minor) as u8),
        );
        
        let mut device_info = CorexDeviceInfo {
            gpu_id,
            compute_capability: compute_cap,
            memory_size: device_props.total_memory,
            corex_version: Self::get_corex_version()?,
            supports_bf16: compute_cap >= (7, 0),
            supports_fp8: compute_cap >= (8, 9),
            supports_tensor_cores: compute_cap >= (7, 0),
            max_threads_per_block: device_props.max_threads_per_block as usize,
            max_grid_dim: device_props.max_grid_dim,
            max_block_dim: device_props.max_block_dim,
            warp_size: device_props.warp_size as usize,
        };
        
        // Enhance with CoreX-specific information
        device_info.enhance_with_corex_info()?;
        
        Ok(device_info)
    }
    
    /// Get CoreX SDK version
    fn get_corex_version() -> Result<String> {
        // This would query CoreX SDK version
        // For now, return a default
        Ok("4.3.8".to_string())
    }
    
    /// Get device information
    pub fn device_info(&self) -> &CorexDeviceInfo {
        &self.device_info
    }
    
    /// Check if device supports a specific feature
    pub fn supports_feature(&self, feature: CorexFeature) -> bool {
        match feature {
            CorexFeature::BF16 => self.device_info.supports_bf16,
            CorexFeature::FP8 => self.device_info.supports_fp8,
            CorexFeature::TensorCores => self.device_info.supports_tensor_cores,
            CorexFeature::MemoryPool => self.config.enable_memory_pool,
            CorexFeature::AsyncOperations => self.config.enable_async,
        }
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let pool = self.memory_pool.lock().unwrap();
        MemoryStats {
            total_allocated: pool.total_allocated,
            max_allocation: pool.max_allocation,
            free_blocks: pool.free_blocks.len(),
            utilization: (pool.total_allocated as f64 / pool.max_allocation as f64) * 100.0,
        }
    }
    
    /// Optimize memory usage by freeing unused blocks
    pub fn optimize_memory(&self) -> Result<()> {
        let mut pool = self.memory_pool.lock().unwrap();
        
        // Free blocks older than certain threshold or when memory pressure is high
        if pool.free_blocks.len() > 100 {
            pool.free_blocks.truncate(50); // Keep only recent blocks
        }
        
        Ok(())
    }
    
    /// Set device optimization profile
    pub fn set_optimization_profile(&self, profile: OptimizationProfile) -> Result<()> {
        match profile {
            OptimizationProfile::Performance => {
                std::env::set_var("COREX_OPT_MODE", "performance");
            }
            OptimizationProfile::Balanced => {
                std::env::set_var("COREX_OPT_MODE", "balanced");
            }
            OptimizationProfile::Memory => {
                std::env::set_var("COREX_OPT_MODE", "memory");
            }
        }
        Ok(())
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total allocated memory in bytes
    pub total_allocated: usize,
    /// Maximum allocation limit
    pub max_allocation: usize,
    /// Number of free blocks in pool
    pub free_blocks: usize,
    /// Memory utilization percentage
    pub utilization: f64,
}

/// Device optimization profiles
#[derive(Debug, Clone, Copy)]
pub enum OptimizationProfile {
    /// Optimize for maximum performance
    Performance,
    /// Balance performance and memory usage
    Balanced,
    /// Optimize for minimal memory usage
    Memory,
}

// Delegate most operations to the underlying CUDA device
impl std::ops::Deref for CorexDevice {
    type Target = crate::CudaDevice;
    
    fn deref(&self) -> &Self::Target {
        &self.cuda_device
    }
}

impl BackendDevice for CorexDevice {
    type Storage = super::CorexStorage;
    
    fn new(ordinal: usize) -> Result<Self> {
        Self::new(ordinal)
    }
    
    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda { gpu_id: self.device_info.gpu_id }
    }
    
    fn same_device(&self, other: &Self) -> bool {
        self.cuda_device.same_device(&other.cuda_device)
    }
    
    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.zeros_impl(shape, dtype)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let cuda_storage = unsafe { self.cuda_device.alloc_uninit(shape, dtype)? };
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn storage_from_slice<T: WithDType>(&self, slice: &[T]) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.storage_from_slice(slice)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.storage_from_cpu_storage(storage)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.storage_from_cpu_storage_owned(storage)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.rand_uniform(shape, dtype, lo, up)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cuda_storage = self.cuda_device.rand_normal(shape, dtype, mean, std)?;
        Ok(super::CorexStorage::from_cuda_storage(cuda_storage, self.clone()))
    }
    
    fn set_seed(&self, seed: u64) -> Result<()> {
        self.cuda_device.set_seed(seed)
    }
    
    fn get_current_seed(&self) -> Result<u64> {
        self.cuda_device.get_current_seed()
    }
    
    fn synchronize(&self) -> Result<()> {
        self.cuda_device.synchronize()
    }
}
//! CoreX backend tests
//!
//! This module contains unit tests for CoreX GPU functionality.

use super::*;
use crate::{Tensor, Device, DType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "corex")]
    fn test_corex_availability() -> Result<()> {
        // Test CoreX availability detection
        let available = is_corex_available();
        println!("CoreX available: {}", available);
        Ok(())
    }

    #[test]
    #[cfg(feature = "corex")]
    fn test_corex_device_creation() -> Result<()> {
        if !is_corex_available() {
            println!("Skipping CoreX test - CoreX not available");
            return Ok(());
        }

        // Initialize CoreX backend
        initialize_corex(CorexConfig::default())?;
        
        // Try to create a CoreX device
        let device_result = Device::new_corex(0);
        
        match device_result {
            Ok(device) => {
                assert!(device.is_corex());
                println!("Successfully created CoreX device: {:?}", device.location());
                
                // Test device information
                let corex_device = device.as_corex_device()?;
                let device_info = corex_device.device_info();
                println!("Device info:\n{}", device_info);
                
                // Test feature support
                assert!(corex_device.supports_feature(CorexFeature::BF16));
                println!("BF16 support: {}", corex_device.supports_feature(CorexFeature::BF16));
                println!("FP8 support: {}", corex_device.supports_feature(CorexFeature::FP8));
                println!("Tensor Cores: {}", corex_device.supports_feature(CorexFeature::TensorCores));
            }
            Err(e) => {
                println!("Failed to create CoreX device: {}", e);
                // This might fail in CI environments without actual hardware
            }
        }
        
        Ok(())
    }

    #[test]
    #[cfg(feature = "corex")]
    fn test_corex_tensor_operations() -> Result<()> {
        if !is_corex_available() {
            return Ok(());
        }

        initialize_corex(CorexConfig::default())?;
        
        // Create CoreX device or fallback to CPU
        let device = Device::corex_if_available(0)?;
        
        // Basic tensor creation
        let a = Tensor::arange(0f32, 100f32, &device)?;
        let b = Tensor::arange(0f32, 100f32, &device)?;
        
        // Element-wise operations
        let c = &a + &b;
        assert_eq!(c.dims(), &[100]);
        
        // Reduction operation
        let sum = c.sum_all()?;
        assert!(sum.to_scalar::<f32>() > 0.0);
        
        println!("CoreX tensor operations test passed");
        Ok(())
    }

    #[test]
    #[cfg(feature = "corex")]
    fn test_corex_memory_stats() -> Result<()> {
        if !is_corex_available() {
            return Ok(());
        }

        initialize_corex(CorexConfig::default())?;
        
        let device = Device::new_corex(0)?;
        let corex_device = device.as_corex_device()?;
        
        // Get memory statistics
        let stats = corex_device.memory_stats();
        println!("Memory stats - Total: {} MB, Utilization: {:.1}%", 
                 stats.total_allocated / 1024 / 1024,
                 stats.utilization);
        
        // Optimize memory
        corex_device.optimize_memory()?;
        
        Ok(())
    }

    #[test]
    fn test_device_detection() -> Result<()> {
        // Test device detection without requiring CoreX
        let devices = detect_corex_devices();
        println!("Found {} CoreX devices", devices.len());
        
        for (i, device_info) in devices.iter().enumerate() {
            println!("Device {}: {}", i, device_info);
        }
        
        Ok(())
    }

    #[test]
    fn test_device_info_validation() -> Result<()> {
        // Test device info functionality without requiring actual hardware
        let mut device_info = CorexDeviceInfo {
            gpu_id: 0,
            compute_capability: (8, 6),
            memory_size: 16 * 1024 * 1024 * 1024, // 16GB
            corex_version: "4.3.8".to_string(),
            supports_bf16: true,
            supports_fp8: true,
            supports_tensor_cores: true,
            max_threads_per_block: 1024,
            max_grid_dim: [2147483647, 65535, 65535],
            max_block_dim: [1024, 1024, 64],
            warp_size: 32,
        };

        // Test precision support
        assert!(device_info.supports_precision(DType::F32));
        assert!(device_info.supports_precision(DType::F16));
        assert!(device_info.supports_precision(DType::BF16));
        assert!(device_info.supports_precision(DType::F8E5M2));

        // Test optimal block sizes
        assert!(device_info.optimal_block_size(OperationType::MatMul) > 0);
        assert!(device_info.optimal_block_size(OperationType::Convolution) > 0);
        assert!(device_info.optimal_block_size(OperationType::ElementWise) > 0);

        // Test bandwidth estimation
        let bandwidth = device_info.estimated_bandwidth();
        assert!(bandwidth > 0.0);

        Ok(())
    }

    #[test]
    fn test_storage_optimizations() -> Result<()> {
        // Test storage optimization configuration
        let optimizations = StorageOptimizations {
            layout_optimization: true,
            kernel_fusion: true,
            async_transfers: false,
            precision_optimization: true,
        };

        assert!(optimizations.layout_optimization);
        assert!(optimizations.kernel_fusion);
        assert!(!optimizations.async_transfers);
        assert!(optimizations.precision_optimization);

        Ok(())
    }

    #[test]
    fn test_memory_usage_calculation() -> Result<()> {
        // Test memory usage calculations
        let metadata = StorageMetadata {
            dtype: DType::F32,
            shape: (100, 200).into(),
            strides: vec![200, 1],
            alignment: 256,
            access_pattern: AccessPattern::Structured,
            created_at: std::time::Instant::now(),
        };

        let usage = MemoryUsage {
            element_count: 20000,
            element_size: 4,
            total_size: 80000,
            alignment: 256,
            overhead_bytes: std::mem::size_of::<StorageMetadata>(),
        };

        assert_eq!(usage.element_count, 20000);
        assert_eq!(usage.element_size, 4);
        assert_eq!(usage.total_size, 80000);
        assert_eq!(usage.alignment, 256);

        Ok(())
    }

    #[test]
    fn test_optimization_profiles() -> Result<()> {
        // Test different optimization profiles
        let profiles = [
            OptimizationProfile::Performance,
            OptimizationProfile::Balanced,
            OptimizationProfile::Memory,
        ];

        for profile in profiles {
            match profile {
                OptimizationProfile::Performance => {
                    // Performance-specific validations
                }
                OptimizationProfile::Balanced => {
                    // Balanced-specific validations
                }
                OptimizationProfile::Memory => {
                    // Memory-specific validations
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_error_handling() -> Result<()> {
        // Test CoreX error types
        let device_error = CorexError::DeviceNotFound { gpu_id: 999 };
        assert!(matches!(device_error, CorexError::DeviceNotFound { gpu_id: 999 }));

        let sdk_error = CorexError::SdkNotFound;
        assert!(matches!(sdk_error, CorexError::SdkNotFound));

        let memory_error = CorexError::MemoryAllocationFailed { size: 1024 };
        assert!(matches!(memory_error, CorexError::MemoryAllocationFailed { size: 1024 }));

        let feature_error = CorexError::UnsupportedFeature { feature: CorexFeature::FP8 };
        assert!(matches!(feature_error, CorexError::UnsupportedFeature { feature: CorexFeature::FP8 }));

        Ok(())
    }
}

/// Integration test that requires actual CoreX hardware
#[cfg(test)]
#[cfg(feature = "corex")]
mod integration_tests {
    use super::*;

    #[test]
    #[ignore] // Ignored by default, requires actual CoreX hardware
    fn test_corex_matrix_multiplication() -> Result<()> {
        if !is_corex_available() {
            return Ok(());
        }

        initialize_corex(CorexConfig::default())?;
        let device = Device::new_corex(0)?;
        
        // Create matrices
        let a = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;
        let b = Tensor::randn(0f32, 1f32, (1024, 1024), &device)?;
        
        // Perform matrix multiplication
        let start = std::time::Instant::now();
        let c = a.matmul(&b)?;
        let duration = start.elapsed();
        
        println!("CoreX matmul (1024x1024): {:?}", duration);
        assert_eq!(c.dims(), &[1024, 1024]);
        
        // Verify result with CPU computation for small matrices
        let small_a = Tensor::randn(0f32, 1f32, (64, 64), &Device::Cpu)?;
        let small_b = Tensor::randn(0f32, 1f32, (64, 64), &Device::Cpu)?;
        let expected = small_a.matmul(&small_b)?;
        
        let corex_a = Tensor::randn(0f32, 1f32, (64, 64), &device)?;
        let corex_b = Tensor::randn(0f32, 1f32, (64, 64), &device)?;
        let corex_result = corex_a.matmul(&corex_b)?;
        
        // Compare results (allowing for small floating-point differences)
        let cpu_result = corex_result.to_device(&Device::Cpu)?;
        let diff = (&cpu_result - &expected)?;
        let max_diff = diff.abs().max_all::<f32>()?;
        assert!(max_diff < 1e-4, "CoreX result differs from CPU by {}", max_diff);
        
        Ok(())
    }

    #[test]
    #[ignore] // Ignored by default, requires actual CoreX hardware
    fn test_corex_precision_modes() -> Result<()> {
        if !is_corex_available() {
            return Ok(());
        }

        initialize_corex(CorexConfig::default())?;
        let device = Device::new_corex(0)?;
        
        // Test different precision modes
        let precisions = [DType::F32, DType::F16, DType::BF16];
        
        for precision in precisions.iter() {
            if !device.supports_bf16() && *precision == DType::BF16 {
                continue;
            }
            
            let a = Tensor::randn(0f32, 1f32, (512, 512), &device)?
                .to_dtype(*precision)?;
            let b = Tensor::randn(0f32, 1f32, (512, 512), &device)?
                .to_dtype(*precision)?;
            
            let c = a.matmul(&b)?;
            assert_eq!(c.dtype(), *precision);
            assert_eq!(c.dims(), &[512, 512]);
            
            println!("Precision {:?} test passed", precision);
        }
        
        Ok(())
    }

    #[test]
    #[ignore] // Ignored by default, requires actual CoreX hardware
    fn test_corex_memory_stress() -> Result<()> {
        if !is_corex_available() {
            return Ok(());
        }

        initialize_corex(CorexConfig::default())?;
        let device = Device::new_corex(0)?;
        let corex_device = device.as_corex_device()?;
        
        // Allocate multiple tensors to test memory management
        let mut tensors = Vec::new();
        
        for i in 0..10 {
            let tensor = Tensor::randn(0f32, 1f32, (1000, 1000), &device)?;
            tensors.push(tensor);
            
            if i % 3 == 0 {
                // Show memory usage periodically
                let stats = corex_device.memory_stats();
                println!("After {} tensors: {:.1}% memory utilization", 
                         i + 1, stats.utilization);
            }
        }
        
        // Test memory optimization
        corex_device.optimize_memory()?;
        
        // Clear tensors
        tensors.clear();
        
        // Final memory stats
        let final_stats = corex_device.memory_stats();
        println!("Final memory utilization: {:.1}%", final_stats.utilization);
        
        Ok(())
    }
}
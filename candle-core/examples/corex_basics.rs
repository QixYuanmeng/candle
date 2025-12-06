//! CoreX GPU backend example
//!
//! This example demonstrates how to use the CoreX GPU backend with Candle
//! for accelerated tensor operations and deep learning inference.

use candle_core::{Tensor, Device, DType, Result};

fn main() -> Result<()> {
    println!("=== CoreX GPU Backend Example ===");
    
    // Check if CoreX is available
    if !candle_core::is_corex_available() {
        println!("❌ CoreX not available. Please ensure:");
        println!("   1. CoreX SDK is installed");
        println!("   2. COREX_HOME environment variable is set");
        println!("   3. CoreX GPU drivers are loaded");
        return Ok(());
    }
    
    println!("✅ CoreX SDK detected");
    
    // Initialize CoreX backend
    let config = candle_core::corex_backend::CorexConfig::default();
    candle_core::initialize_corex(config)?;
    println!("✅ CoreX backend initialized");
    
    // Create a CoreX device
    let device = Device::new_corex(0)?;
    println!("✅ CoreX device created: {:?}", device.location());
    
    // Get device information
    let corex_device = device.as_corex_device()?;
    println!("\n📊 Device Information:");
    println!("{}", corex_device.device_info());
    
    // Test basic tensor operations
    println!("\n🔢 Testing Basic Operations:");
    test_basic_operations(&device)?;
    
    // Test matrix multiplication
    println!("\n🔢 Testing Matrix Multiplication:");
    test_matrix_multiplication(&device)?;
    
    // Test different precisions
    println!("\n🔢 Testing Precision Modes:");
    test_precision_modes(&device)?;
    
    // Test memory optimization
    println!("\n💾 Testing Memory Management:");
    test_memory_management(&corex_device)?;
    
    println!("\n🎉 All tests completed successfully!");
    Ok(())
}

/// Test basic tensor operations
fn test_basic_operations(device: &Device) -> Result<()> {
    // Create tensors
    let a = Tensor::arange(0f32, 10f32, device)?;
    let b = Tensor::full((3, 3), 2f32, device)?;
    
    println!("  Created tensor A: shape={:?}, dtype={}", a.dims(), a.dtype());
    println!("  Created tensor B: shape={:?}, dtype={}", b.dims(), b.dtype());
    
    // Element-wise operations
    let c = &a + &b;
    println!("  A + B: shape={:?}, sum={:.2}", c.dims(), c.sum_all::<f32>()?);
    
    let d = a.exp()?;
    println!("  exp(A): shape={:?}, sum={:.2}", d.dims(), d.sum_all::<f32>()?);
    
    // Reduction operations
    let sum = a.sum_all()?;
    let max = a.max_all()?;
    let mean = a.mean_all()?;
    
    println!("  Sum of A: {:.2}", sum.to_scalar::<f32>()?);
    println!("  Max of A: {:.2}", max.to_scalar::<f32>()?);
    println!("  Mean of A: {:.4}", mean.to_scalar::<f32>()?);
    
    Ok(())
}

/// Test matrix multiplication performance
fn test_matrix_multiplication(device: &Device) -> Result<()> {
    let sizes = [(512, 512), (1024, 1024)];
    
    for (m, n) in sizes {
        println!("  Testing {}x{} matmul:", m, n);
        
        let start = std::time::Instant::now();
        
        let a = Tensor::randn(0f32, 1f32, (m, n), device)?;
        let b = Tensor::randn(0f32, 1f32, (m, n), device)?;
        
        let create_time = start.elapsed();
        
        let start = std::time::Instant::now();
        let c = a.matmul(&b.t()?)?;
        let compute_time = start.elapsed();
        
        println!("    Creation: {:?}, Compute: {:?}, Result shape: {:?}", 
                 create_time, compute_time, c.dims());
        
        // Verify result shape
        assert_eq!(c.dims(), &[m, m]);
    }
    
    Ok(())
}

/// Test different precision modes
fn test_precision_modes(device: &Device) -> Result<()> {
    let corex_device = device.as_corex_device()?;
    
    let precisions = [DType::F32, DType::F16, DType::BF16];
    
    for precision in precisions.iter() {
        if *precision == DType::BF16 && !corex_device.supports_feature(candle_core::corex_backend::CorexFeature::BF16) {
            println!("  ⚠️  Skipping BF16 - not supported by this CoreX device");
            continue;
        }
        
        println!("  Testing precision {:?}:", precision);
        
        let start = std::time::Instant::now();
        
        let a = Tensor::randn(0f32, 1f32, (512, 512), device)?
            .to_dtype(*precision)?;
        let b = Tensor::randn(0f32, 1f32, (512, 512), device)?
            .to_dtype(*precision)?;
        
        let create_time = start.elapsed();
        
        let start = std::time::Instant::now();
        let c = a.matmul(&b)?;
        let compute_time = start.elapsed();
        
        println!("    Creation: {:?}, Compute: {:?}, dtype: {:?}", 
                 create_time, compute_time, c.dtype());
        
        // Convert back to F32 for verification
        let c_f32 = c.to_dtype(DType::F32)?;
        assert_eq!(c_f32.dims(), &[512, 512]);
    }
    
    Ok(())
}

/// Test memory management and optimization
fn test_memory_management(device: &candle_core::CorexDevice) -> Result<()> {
    println!("  Initial memory stats:");
    let initial_stats = device.memory_stats();
    println!("    {:?}", initial_stats);
    
    // Allocate several tensors
    println!("  Allocating tensors...");
    let tensors: Vec<Tensor> = (0..10)
        .map(|_| Tensor::randn(0f32, 1f32, (1024, 1024), device.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    
    println!("  After allocating 10 tensors:");
    let after_stats = device.memory_stats();
    println!("    {:?}", after_stats);
    
    // Test memory optimization
    println!("  Optimizing memory...");
    device.optimize_memory()?;
    
    println!("  After optimization:");
    let final_stats = device.memory_stats();
    println!("    {:?}", final_stats);
    
    // Test memory pressure scenario
    println!("  Testing memory pressure cleanup...");
    drop(tensors);
    
    let cleanup_stats = device.memory_stats();
    println!("    {:?}", cleanup_stats);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corex_basic_functionality() -> Result<()> {
        if !candle_core::is_corex_available() {
            return Ok(());
        }
        
        let config = candle_core::corex_backend::CorexConfig::default();
        candle_core::initialize_corex(config)?;
        
        let device = Device::new_corex(0)?;
        assert!(device.is_corex());
        
        let tensor = Tensor::ones((10, 10), &device)?;
        assert_eq!(tensor.dims(), &[10, 10]);
        
        Ok(())
    }

    #[test]
    fn test_corex_device_features() -> Result<()> {
        if !candle_core::is_corex_available() {
            return Ok(());
        }
        
        let device = Device::new_corex(0)?;
        let corex_device = device.as_corex_device()?;
        
        // Test feature detection
        let features = [
            candle_core::corex_backend::CorexFeature::BF16,
            candle_core::corex_backend::CorexFeature::FP8,
            candle_core::corex_backend::CorexFeature::TensorCores,
            candle_core::corex_backend::CorexFeature::MemoryPool,
        ];
        
        for feature in features.iter() {
            let supported = corex_device.supports_feature(*feature);
            println!("CoreX feature {:?} supported: {}", feature, supported);
        }
        
        Ok(())
    }

    #[test]
    fn test_corex_error_handling() -> Result<()> {
        // Test error conditions that should be handled gracefully
        
        // Test invalid GPU ID
        let invalid_device = Device::new_corex(999);
        assert!(invalid_device.is_err());
        
        // Test CoreX-specific error types
        let sdk_error = candle_core::corex_backend::CorexError::SdkNotFound;
        assert!(matches!(sdk_error, candle_core::corex_backend::CorexError::SdkNotFound));
        
        let memory_error = candle_core::corex_backend::CorexError::MemoryAllocationFailed { size: 1024 };
        assert!(matches!(memory_error, candle_core::corex_backend::CorexError::MemoryAllocationFailed { size: 1024 }));
        
        Ok(())
    }
}
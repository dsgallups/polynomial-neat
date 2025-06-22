# Migration from Candle to Burn

This document describes the migration of the polynomial-neat project from the Candle deep learning framework to the Burn framework.

## Overview

The project has been successfully migrated from Candle to Burn while maintaining all functionality. Burn was chosen as it provides similar tensor operations with a more modern and flexible architecture.

## Major Changes

### 1. Package and Dependencies

**Before:**
```toml
[package]
name = "candle-neat"

[dependencies]
candle-nn = "0.9.1"
```

**After:**
```toml
[package]
name = "polynomial-neat"

[dependencies]
burn = { version = "0.17.1", features = ["ndarray", "wgpu"] }
```

### 2. Module Structure

- The `candle_net` module has been replaced with `burn_net`
- The `candle_net` module is commented out in `poly/mod.rs` to avoid compilation errors
- All Candle-specific code has been ported to use Burn's API

### 3. Backend and Device Changes

**Candle:**
```rust
use candle_core::Device;
let dev = Device::new_metal(0).unwrap();
```

**Burn:**
```rust
use burn::backend::Wgpu;
let device = burn::backend::wgpu::WgpuDevice::default();
```

For testing with NdArray backend:
```rust
use burn::backend::NdArray;
type TestBackend = NdArray;
let device = burn::backend::ndarray::NdArrayDevice::default();
```

### 4. Tensor Operations

**Creating tensors:**

Candle:
```rust
use candle_core::{Device, Tensor};
let tensor = Tensor::new(values, &Device::Cpu)?;
```

Burn:
```rust
use burn::prelude::*;
let data = TensorData::new(values, shape);
let tensor = Tensor::from_data(data, &device);
```

**Tensor reshaping:**

Candle:
```rust
tensor.reshape((rows, cols))?
```

Burn:
```rust
tensor.reshape([rows, cols])
```

**Matrix multiplication:**

Candle:
```rust
tensor1.matmul(&tensor2)?
```

Burn:
```rust
tensor1.clone().matmul(tensor2)  // Note: Burn takes ownership
```

### 5. Data Extraction

**Candle:**
```rust
let data = tensor.to_vec1()?;
```

**Burn:**
```rust
let data = tensor.to_data();
let vec = data.as_slice::<f32>().unwrap().to_vec();
```

### 6. Import Updates

All imports have been updated:

**Before:**
```rust
use candle_neat::{activated::prelude::*, topology::mutation::MutationChances};
```

**After:**
```rust
use polynomial_neat::{activated::prelude::*, topology::mutation::MutationChances};
```

### 7. Error Handling

Burn operations don't return `Result` types like Candle, making the API cleaner:

**Candle:**
```rust
let result = tensor.operation()?.another_operation()?;
```

**Burn:**
```rust
let result = tensor.operation().another_operation();
```

## File Changes Summary

1. **Cargo.toml**: Updated package name and dependencies
2. **src/main.rs**: Updated to use BurnNetwork instead of CandleNetwork
3. **src/poly/mod.rs**: Added burn_net module, commented out candle_net
4. **src/poly/burn_net/**: New module with Burn implementations
   - `mod.rs`: Main module file
   - `basis_prime.rs`: Tensor creation using Burn
   - `coeff.rs`: Coefficient tensor handling
   - `network.rs`: Main network implementation
   - `tests.rs`: Test suite
   - `expander/`: Copied from candle_net (device-agnostic)
5. **src/scratch.rs**: Updated to use Burn tensors
6. **src/poly/tests.rs**: Updated to use BurnNetwork
7. **src/activated/main.rs**: Updated package import

## Testing

All tests have been ported and are passing:
- `cargo test --lib burn_scratch`
- `cargo test --lib test_burn_network_functionality`

The project builds successfully with `cargo build --release`.

## Performance Considerations

The Burn framework offers multiple backends:
- **NdArray**: CPU-based, good for testing and development
- **WGPU**: GPU acceleration via WebGPU, cross-platform
- **CUDA**: NVIDIA GPU acceleration (requires additional features)

The current implementation uses WGPU for the main application and NdArray for tests.

## Future Improvements

1. The `candle_net` module could be removed entirely once the migration is fully validated
2. Additional Burn backends (CUDA, ROCm) could be explored for better performance
3. Burn's autodiff capabilities could be leveraged for training scenarios

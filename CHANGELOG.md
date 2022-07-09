# CUDA Extension Changelog
All notable changes to the Neighborhood Attention CUDA extension will be documented in this file.
 
## [0.12] - 2022-07-09
 
### Added
- Fixed the race condition in K-Backward and V-Backward kernels.
  - This was handled previously with atomic adds (non-deterministic).
  - Now the kernels compute inverse neighbors and compute gradients without a race condition.
- "Tiled" Neighborhood Attention kernels for 5x5, 7x7, 9x9, 11x11, and 13x13 window sizes.
  - Applies only to QKRPB-Forward and A-Backward.
  - Only supports dim per head = 32 for now.
    - Try to keep your channels a multiple of 32.
- Improved FP16 support.
  - Specific kernels for FP16 that use `half2` addressing.
  - Different threads per block for FP16.
- New 1D NA kernels
  - Slightly more efficient.
  - With FP16 support.
  - Arbitrary kernel sizes supported (there's no upper bound of 13x13 like the 2D version).
 
### Changed
- Window size templating, and some other common bits are factored out into a `commons` header file.
  - Makes extending support and editing easier.
- Gradchecks now run in fast mode (faster and less memory usage) unless instructed otherwise.
 
## [0.11a] - 2022-05-12
 
### Added
- 1D Neighborhood Attention
 
## [0.11] - 2022-04-30
 
### Changed
  
- Classification and downstream kernels combined.
- Refactored cuda extension to `natten/`.
- Minor speed improvements.
 
## [0.10]
Initial public release.
// Batched tiny GEMM with FP8 storage and FP32 accumulation (docs/SPEC.md).
// One warp per matrix, register-resident: lane stripes over output elements,
// converts FP8->FP32 on the fly, accumulates the row.col dot in an FP32
// register. CUDA cores only in v1 — no tensor-core mma; the bet being tested
// is launch/setup tax, not FLOPs. Row-major A, B, C, batch-major layout.
#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace ftg {

// Quantize on the host with round-to-nearest-even and finite saturation.
inline std::uint8_t quantize_fp8(float x, __nv_fp8_interpretation_t interp) {
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, interp));
}

inline std::uint8_t quantize_fp4(float x) {
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest));
}

__device__ inline float dequant_fp8(std::uint8_t v, __nv_fp8_interpretation_t interp) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, interp)));
}

__device__ inline float dequant_fp4(std::uint8_t v) {
  return __half2float(__half(__nv_cvt_fp4_to_halfraw(v, __NV_E2M1)));
}

template <__nv_fp8_interpretation_t INTERP>
__global__ void gemm_fp8(const std::uint8_t* __restrict__ A,
                         const std::uint8_t* __restrict__ B,
                         float* __restrict__ C, int n, int batch) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  const std::size_t base = static_cast<std::size_t>(b) * n * n;
  const std::uint8_t* a = A + base;
  const std::uint8_t* bm = B + base;
  float* c = C + base;
  for (int idx = threadIdx.x; idx < n * n; idx += 32) {
    const int r = idx / n, col = idx % n;
    float acc = 0.0f;
    for (int k = 0; k < n; ++k)
      acc += dequant_fp8(a[r * n + k], INTERP) * dequant_fp8(bm[k * n + col], INTERP);
    c[idx] = acc;
  }
}

// FP4 e2m1: accuracy appendix only (docs/SPEC.md) — never a performance bet.
__global__ void gemm_fp4(const std::uint8_t* __restrict__ A,
                         const std::uint8_t* __restrict__ B,
                         float* __restrict__ C, int n, int batch) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  const std::size_t base = static_cast<std::size_t>(b) * n * n;
  const std::uint8_t* a = A + base;
  const std::uint8_t* bm = B + base;
  float* c = C + base;
  for (int idx = threadIdx.x; idx < n * n; idx += 32) {
    const int r = idx / n, col = idx % n;
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) acc += dequant_fp4(a[r * n + k]) * dequant_fp4(bm[k * n + col]);
    c[idx] = acc;
  }
}

}  // namespace ftg

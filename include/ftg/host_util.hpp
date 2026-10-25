// Host-side helpers shared by the tests and benchmark: fixed-seed data,
// FP8/FP4 quantization, the FP64 reference GEMM, and the relative L2 error.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include "ftg.cuh"

namespace ftg {

inline void fill_random(std::vector<float>& v, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (auto& x : v) x = u(rng);
}

inline std::vector<std::uint8_t> quantize(const std::vector<float>& v,
                                          __nv_fp8_interpretation_t interp) {
  std::vector<std::uint8_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = quantize_fp8(v[i], interp);
  return out;
}

inline std::vector<std::uint8_t> quantize_fp4_all(const std::vector<float>& v) {
  std::vector<std::uint8_t> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = quantize_fp4(v[i]);
  return out;
}

// FP64 reference over the original (unquantized) values: the measured error
// therefore includes input quantization, which is the dominant term.
inline void gemm_fp64_ref(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<double>& C, int n, int batch) {
  C.assign(static_cast<std::size_t>(batch) * n * n, 0.0);
  for (int b = 0; b < batch; ++b) {
    const std::size_t base = static_cast<std::size_t>(b) * n * n;
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k)
          acc += static_cast<double>(A[base + r * n + k]) * B[base + k * n + c];
        C[base + r * n + c] = acc;
      }
  }
}

inline double relative_l2(const std::vector<float>& got, const std::vector<double>& ref) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

}  // namespace ftg

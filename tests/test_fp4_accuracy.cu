// FP4 e2m1 accuracy appendix (docs/SPEC.md): accuracy only, no vendor
// baseline, no performance claim. e2m1 has one explicit mantissa bit
// (u = 2^-2) vs e4m3's three (u = 2^-4), for a naive mantissa-count
// prediction of 4x e4m3's error. Measured is ~11x: on U(-1,1) data the
// e2m1 grid rounds every input to {0, +-0.5, +-1}, so a quarter of the
// inputs become exact zeros and the error is absolute-grid dominated, not
// relative-roundoff dominated. Asserted inside [8, 16].

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "ftg/host_util.hpp"

static void cu(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error at %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

int main() {
  int failures = 0;
  for (int n : {16, 32, 64}) {
    const int batch = 64;
    std::vector<float> A(batch * n * n), B(batch * n * n);
    ftg::fill_random(A, 300 + n);
    ftg::fill_random(B, 400 + n);
    std::vector<double> ref;
    ftg::gemm_fp64_ref(A, B, ref, n, batch);

    std::vector<std::uint8_t> A4 = ftg::quantize_fp4_all(A);
    std::vector<std::uint8_t> B4 = ftg::quantize_fp4_all(B);
    std::vector<std::uint8_t> A8 = ftg::quantize(A, __NV_E4M3);
    std::vector<std::uint8_t> B8 = ftg::quantize(B, __NV_E4M3);

    std::uint8_t *dA, *dB;
    float* dC;
    const std::size_t bytes = A4.size();
    cu(cudaMalloc(&dA, bytes), "malloc A");
    cu(cudaMalloc(&dB, bytes), "malloc B");
    cu(cudaMalloc(&dC, bytes * sizeof(float)), "malloc C");

    std::vector<float> C(batch * n * n);
    cu(cudaMemcpy(dA, A4.data(), bytes, cudaMemcpyHostToDevice), "up A4");
    cu(cudaMemcpy(dB, B4.data(), bytes, cudaMemcpyHostToDevice), "up B4");
    ftg::gemm_fp4<<<batch, 32>>>(dA, dB, dC, n, batch);
    cu(cudaGetLastError(), "fp4 launch");
    cu(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost), "down");
    const double rel_fp4 = ftg::relative_l2(C, ref);

    cu(cudaMemcpy(dA, A8.data(), bytes, cudaMemcpyHostToDevice), "up A8");
    cu(cudaMemcpy(dB, B8.data(), bytes, cudaMemcpyHostToDevice), "up B8");
    ftg::gemm_fp8<__NV_E4M3><<<batch, 32>>>(dA, dB, dC, n, batch);
    cu(cudaGetLastError(), "fp8 launch");
    cu(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost), "down");
    const double rel_fp8 = ftg::relative_l2(C, ref);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    const double ratio = rel_fp4 / rel_fp8;
    std::printf("n=%d e2m1 rel=%.4e e4m3 rel=%.4e ratio=%.2fx\n", n, rel_fp4, rel_fp8, ratio);
    if (ratio < 8.0 || ratio > 16.0) {
      std::fprintf(stderr, "FAIL: e2m1/e4m3 ratio %.2f outside [8, 16]\n", ratio);
      ++failures;
    }
  }
  if (failures != 0) return 1;
  std::puts("fp4 appendix: all checks passed");
  return 0;
}

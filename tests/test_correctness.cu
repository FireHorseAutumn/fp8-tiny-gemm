// Correctness vs an FP64 oracle (docs/SPEC.md). Error-bound derivation:
// e4m3 has 3 mantissa bits, so quantization unit roundoff u = 2^-4; each
// quantized product a_k*b_k carries relative error ~ 2u worst-case, and the
// K-term dot random-walks to ~ u*sqrt(K)*||a||*||b||. Assert the measured
// relative L2 error within 3x that bound and report the actual ratio.
// e5m2 has 2 mantissa bits (u = 2^-3): expect ~2x worse, asserted in
// [1.5, 3]. The cuBLASLt baseline is checked against the same bound so the
// benchmark's baseline is known-good.

#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "ftg/host_util.hpp"
#include "ftg/cublaslt_wrap.hpp"

static int failures = 0;

static void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

static void cu(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error at %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

struct RunResult {
  double rel;
  int nonfinite;
};

template <__nv_fp8_interpretation_t INTERP>
static RunResult run_kernel(const std::vector<std::uint8_t>& Aq,
                            const std::vector<std::uint8_t>& Bq,
                            const std::vector<double>& ref, int n, int batch) {
  std::uint8_t *dA, *dB;
  float* dC;
  const std::size_t bytes = Aq.size();
  cu(cudaMalloc(&dA, bytes), "malloc A");
  cu(cudaMalloc(&dB, bytes), "malloc B");
  cu(cudaMalloc(&dC, bytes * sizeof(float)), "malloc C");
  cu(cudaMemcpy(dA, Aq.data(), bytes, cudaMemcpyHostToDevice), "up A");
  cu(cudaMemcpy(dB, Bq.data(), bytes, cudaMemcpyHostToDevice), "up B");
  ftg::gemm_fp8<INTERP><<<batch, 32>>>(dA, dB, dC, n, batch);
  cu(cudaGetLastError(), "launch");
  cu(cudaDeviceSynchronize(), "sync");
  std::vector<float> C(batch * n * n);
  cu(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost), "down C");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  RunResult r;
  r.rel = ftg::relative_l2(C, ref);
  r.nonfinite = 0;
  for (float v : C)
    if (!std::isfinite(v)) ++r.nonfinite;
  return r;
}

int main() {
  const double u4 = 0.0625;  // 2^-4, e4m3
  const double u5 = 0.125;   // 2^-3, e5m2
  void* workspace;
  cu(cudaMalloc(&workspace, 32u << 20), "workspace");

  for (int n : {16, 32, 64}) {
    const int batch = 64;
    std::vector<float> A(batch * n * n), B(batch * n * n);
    ftg::fill_random(A, 100 + n);
    ftg::fill_random(B, 200 + n);
    std::vector<double> ref;
    ftg::gemm_fp64_ref(A, B, ref, n, batch);

    const RunResult r4 = run_kernel<__NV_E4M3>(ftg::quantize(A, __NV_E4M3),
                                               ftg::quantize(B, __NV_E4M3), ref, n, batch);
    const RunResult r5 = run_kernel<__NV_E5M2>(ftg::quantize(A, __NV_E5M2),
                                               ftg::quantize(B, __NV_E5M2), ref, n, batch);
    const double bound4 = 3.0 * u4 * std::sqrt(static_cast<double>(n));
    const double bound5 = 3.0 * u5 * std::sqrt(static_cast<double>(n));
    const double ratio = r4.rel / (u4 * std::sqrt(static_cast<double>(n)));
    const double ratio5 = r5.rel / (u5 * std::sqrt(static_cast<double>(n)));
    const double rel_ratio = r5.rel / r4.rel;
    std::printf("n=%d e4m3 rel=%.4e (ratio %.3f) e5m2 rel=%.4e (ratio %.3f, x%.2f)\n", n,
                r4.rel, ratio, r5.rel, ratio5, rel_ratio);
    check(r4.rel <= bound4, "e4m3 within 3x first-principles bound");
    check(r5.rel <= bound5, "e5m2 within 3x first-principles bound");
    check(r4.nonfinite == 0 && r5.nonfinite == 0, "no non-finite outputs");
    check(rel_ratio >= 1.5 && rel_ratio <= 3.0, "e5m2 ~2x worse than e4m3");

    // Baseline sanity: cuBLASLt e4m3 against the same reference and bound.
    std::vector<std::uint8_t> Aq = ftg::quantize(A, __NV_E4M3);
    std::vector<std::uint8_t> Bq = ftg::quantize(B, __NV_E4M3);
    std::uint8_t *dA, *dB;
    float* dC;
    cu(cudaMalloc(&dA, Aq.size()), "lt malloc A");
    cu(cudaMalloc(&dB, Bq.size()), "lt malloc B");
    cu(cudaMalloc(&dC, Aq.size() * sizeof(float)), "lt malloc C");
    cu(cudaMemcpy(dA, Aq.data(), Aq.size(), cudaMemcpyHostToDevice), "lt up A");
    cu(cudaMemcpy(dB, Bq.data(), Bq.size(), cudaMemcpyHostToDevice), "lt up B");
    {
      ftg::LtBatchedFp8 lt(CUDA_R_8F_E4M3, n, batch, workspace, 32u << 20);
      lt.run(dA, dB, dC);
    }
    cu(cudaDeviceSynchronize(), "lt sync");
    std::vector<float> Cl(batch * n * n);
    cu(cudaMemcpy(Cl.data(), dC, Cl.size() * sizeof(float), cudaMemcpyDeviceToHost), "lt down");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    const double rel_lt = ftg::relative_l2(Cl, ref);
    check(rel_lt <= bound4, "cuBLASLt e4m3 within the same bound");

    std::printf("n=%d e4m3 rel=%.4e (ratio %.3f) e5m2 rel=%.4e (ratio %.3f, x%.2f) "
                "cublaslt rel=%.4e\n",
                n, r4.rel, ratio, r5.rel, ratio5, ratio5 / ratio, rel_lt);
  }

  // Degenerate inputs: all-zeros and saturating magnitudes.
  {
    const int n = 32, batch = 4;
    std::vector<float> Z(batch * n * n, 0.0f), ZB(batch * n * n, 0.0f);
    std::vector<double> refz;
    ftg::gemm_fp64_ref(Z, ZB, refz, n, batch);
    const RunResult rz = run_kernel<__NV_E4M3>(ftg::quantize(Z, __NV_E4M3),
                                               ftg::quantize(ZB, __NV_E4M3), refz, n, batch);
    check(rz.rel == 0.0 && rz.nonfinite == 0, "all-zeros gives exact zeros");

    std::vector<float> Big(batch * n * n), BigB(batch * n * n);
    ftg::fill_random(Big, 999);
    ftg::fill_random(BigB, 998);
    int saturated = 0;
    for (auto& v : Big) {
      v *= 900.0f;  // beyond e4m3 max 448 -> saturates
      if (std::fabs(v) > 448.0f) ++saturated;
    }
    std::vector<double> refb;
    ftg::gemm_fp64_ref(Big, BigB, refb, n, batch);
    const RunResult rb = run_kernel<__NV_E4M3>(ftg::quantize(Big, __NV_E4M3),
                                               ftg::quantize(BigB, __NV_E4M3), refb, n, batch);
    check(rb.nonfinite == 0, "saturating inputs: no NaN/Inf in output");
    check(saturated > 0, "saturation case actually saturates");
    std::printf("degenerate: zeros exact; saturated inputs %d/%zu, outputs finite\n", saturated,
                Big.size());
  }

  cudaFree(workspace);
  if (failures != 0) {
    std::fprintf(stderr, "%d correctness checks failed\n", failures);
    return 1;
  }
  std::puts("correctness: all checks passed");
  return 0;
}

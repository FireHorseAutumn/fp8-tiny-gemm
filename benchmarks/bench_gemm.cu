// Benchmark: hand-written kernel vs cuBLASLt batched FP8 (docs/SPEC.md).
// Median-of-21 with [min,max] spreads, raw per-rep timings preserved, µs
// per batched call, effective TFLOP/s and GB/s, plus a batch=1
// launch-overhead exhibit. The kernel uses CUDA cores only — the bet under
// test is launch/setup tax, not FLOPs. JSON to data/fp8_bench.json.

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include "ftg/host_util.hpp"
#include "ftg/cublaslt_wrap.hpp"

namespace {

constexpr int kWarmup = 3;
constexpr int kReps = 21;

void cu(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "cuda error at %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

struct Timed {
  double median_us, min_us, max_us;
  std::vector<double> reps_us;
};

Timed summarize(std::vector<double> reps) {
  std::vector<double> s = reps;
  std::sort(s.begin(), s.end());
  return {s[s.size() / 2], s.front(), s.back(), std::move(reps)};
}

template <typename F>
Timed time_calls(F&& fn) {
  cudaEvent_t ev0, ev1;
  cu(cudaEventCreate(&ev0), "ev");
  cu(cudaEventCreate(&ev1), "ev");
  for (int i = 0; i < kWarmup; ++i) fn();
  cu(cudaDeviceSynchronize(), "warmup sync");
  std::vector<double> reps;
  for (int i = 0; i < kReps; ++i) {
    cu(cudaEventRecord(ev0), "rec0");
    fn();
    cu(cudaEventRecord(ev1), "rec1");
    cu(cudaEventSynchronize(ev1), "sync");
    float ms = 0.0f;
    cu(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
    reps.push_back(ms * 1e3);
  }
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  return summarize(std::move(reps));
}

void json_reps(FILE* f, const std::vector<double>& r) {
  std::fprintf(f, "[");
  for (std::size_t i = 0; i < r.size(); ++i) std::fprintf(f, "%s%.3f", i ? "," : "", r[i]);
  std::fprintf(f, "]");
}

}  // namespace

int main() {
  cudaDeviceProp prop;
  cu(cudaGetDeviceProperties(&prop, 0), "props");
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);

  void* workspace;
  cu(cudaMalloc(&workspace, 32u << 20), "workspace");

  FILE* f = std::fopen("data/fp8_bench.json", "w");
  if (!f) {
    std::fprintf(stderr, "cannot open data/fp8_bench.json (run from repo root)\n");
    return 2;
  }
  std::fprintf(f, "{\n\"device\": \"%s\",\n\"compute_capability\": \"%d.%d\",\n"
                  "\"driver\": %d,\n\"cuda_runtime\": %d,\n\"timestamp\": %lld,\n"
                  "\"governor_caveat\": \"laptop, power mode not pinned for this run\",\n"
                  "\"warmup\": %d,\n\"reps\": %d,\n\"results\": [\n",
               prop.name, prop.major, prop.minor, driver, runtime,
               static_cast<long long>(std::time(nullptr)), kWarmup, kReps);

  bool first = true;
  for (int n : {16, 32, 64}) {
    for (std::int64_t batch : {1, 64, 1024, 65536}) {
      for (int kind = 0; kind < 2; ++kind) {
        const __nv_fp8_interpretation_t interp = kind == 0 ? __NV_E4M3 : __NV_E5M2;
        const char* dtype = kind == 0 ? "e4m3" : "e5m2";
        const cudaDataType_t lt_type = kind == 0 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
        const std::size_t elems = static_cast<std::size_t>(batch) * n * n;
        std::vector<float> A(elems), B(elems);
        ftg::fill_random(A, 1000 + n * 7 + batch);
        ftg::fill_random(B, 2000 + n * 7 + batch);
        const std::vector<std::uint8_t> Aq = ftg::quantize(A, interp);
        const std::vector<std::uint8_t> Bq = ftg::quantize(B, interp);

        std::uint8_t *dA, *dB;
        float* dC;
        cu(cudaMalloc(&dA, elems), "malloc A");
        cu(cudaMalloc(&dB, elems), "malloc B");
        cu(cudaMalloc(&dC, elems * sizeof(float)), "malloc C");
        cu(cudaMemcpy(dA, Aq.data(), elems, cudaMemcpyHostToDevice), "up A");
        cu(cudaMemcpy(dB, Bq.data(), elems, cudaMemcpyHostToDevice), "up B");

        Timed kern;
        if (kind == 0)
          kern = time_calls([&] {
            ftg::gemm_fp8<__NV_E4M3><<<static_cast<unsigned>(batch), 32>>>(
                dA, dB, dC, n, static_cast<int>(batch));
            cu(cudaGetLastError(), "kernel launch");
          });
        else
          kern = time_calls([&] {
            ftg::gemm_fp8<__NV_E5M2><<<static_cast<unsigned>(batch), 32>>>(
                dA, dB, dC, n, static_cast<int>(batch));
            cu(cudaGetLastError(), "kernel launch");
          });

        const bool lt_ok = ftg::lt_fp8_supported(lt_type, n, batch);
        Timed vendor{0.0, 0.0, 0.0, {}};
        if (lt_ok) {
          ftg::LtBatchedFp8 lt(lt_type, n, batch, workspace, 32u << 20);
          vendor = time_calls([&] { lt.run(dA, dB, dC); });
        }

        const double flops = 2.0 * n * n * n * static_cast<double>(batch);
        const double bytes = elems * 2 + elems * sizeof(float);
        const char* winner = !lt_ok ? "kernel"
                             : (kern.median_us <= vendor.median_us ? "kernel" : "cublaslt");
        if (!first) std::fprintf(f, ",\n");
        first = false;
        std::fprintf(f,
                     "{\"n\": %d, \"batch\": %lld, \"dtype\": \"%s\", "
                     "\"kernel_us\": %.3f, \"kernel_min\": %.3f, \"kernel_max\": %.3f, ",
                     n, static_cast<long long>(batch), dtype, kern.median_us, kern.min_us,
                     kern.max_us);
        if (lt_ok)
          std::fprintf(f,
                       "\"cublaslt_us\": %.3f, \"cublaslt_min\": %.3f, \"cublaslt_max\": %.3f, "
                       "\"kernel_tflops\": %.4f, \"cublaslt_tflops\": %.4f, "
                       "\"kernel_gbs\": %.2f, \"cublaslt_gbs\": %.2f, \"winner\": \"%s\", ",
                       vendor.median_us, vendor.min_us, vendor.max_us, flops / kern.median_us / 1e6,
                       flops / vendor.median_us / 1e6, bytes / kern.median_us / 1e3,
                       bytes / vendor.median_us / 1e3, winner);
        else
          std::fprintf(f,
                       "\"cublaslt_us\": null, \"cublaslt_note\": \"no algo (unsupported)\", "
                       "\"kernel_tflops\": %.4f, \"kernel_gbs\": %.2f, \"winner\": \"%s\", ",
                       flops / kern.median_us / 1e6, bytes / kern.median_us / 1e3, winner);
        std::fprintf(f, "\"kernel_reps_us\": ");
        json_reps(f, kern.reps_us);
        if (lt_ok) {
          std::fprintf(f, ", \"cublaslt_reps_us\": ");
          json_reps(f, vendor.reps_us);
        }
        std::fprintf(f, "}");

        if (lt_ok)
          std::printf("n=%d batch=%lld %s kernel=%.1fus cublaslt=%.1fus -> %s\n", n,
                      static_cast<long long>(batch), dtype, kern.median_us, vendor.median_us,
                      winner);
        else
          std::printf("n=%d batch=%lld %s kernel=%.1fus cublaslt=unsupported\n", n,
                      static_cast<long long>(batch), dtype, kern.median_us);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
      }
    }
  }
  std::fprintf(f, "\n]\n}\n");
  std::fclose(f);
  cudaFree(workspace);
  std::puts("wrote data/fp8_bench.json");
  return 0;
}

// cuBLASLt batched FP8 baseline in its best honest configuration
// (docs/SPEC.md): strided-batched descriptors, CUBLAS_COMPUTE_32F, FP32
// output, algo chosen by heuristic query (not the default). Row-major
// layouts map to the TN arrangement the FP8 path requires.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>

namespace ftg {

inline void lt_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasLt error at %s: %d\n", what, static_cast<int>(s));
    std::exit(2);
  }
}

struct LtBatchedFp8 {
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
  cublasLtMatmulHeuristicResult_t algo;
  void* workspace = nullptr;
  std::size_t workspace_size = 0;

  // Descriptors are built once per (dtype, n, batch); the timed call is
  // cublasLtMatmul alone.
  LtBatchedFp8(cudaDataType_t dt, int n, std::int64_t batch, void* ws, std::size_t ws_size)
      : workspace(ws), workspace_size(ws_size) {
    lt_check(cublasLtCreate(&lt), "create");
    lt_check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    lt_check(cublasLtMatmulPreferenceCreate(&pref), "pref");
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &ws_size, sizeof(ws_size)),
             "pref ws");
    auto make_layout = [&](cublasLtMatrixLayout_t* lay, cudaDataType_t t) {
      lt_check(cublasLtMatrixLayoutCreate(lay, t, n, n, n), "layout");
      const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
      lt_check(cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                                sizeof(order)),
               "order");
      const int batch_i32 = static_cast<int>(batch);
      lt_check(cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                &batch_i32, sizeof(batch_i32)),
               "batch count");
      const std::int64_t stride = static_cast<std::int64_t>(n) * n;
      lt_check(cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                &stride, sizeof(stride)),
               "stride");
    };
    make_layout(&Ad, dt);
    make_layout(&Bd, dt);
    make_layout(&Cd, CUDA_R_32F);
    int count = 0;
    lt_check(cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &algo, &count),
             "heuristic");
    if (count == 0) {
      std::fprintf(stderr, "cublasLt heuristic: no algo for this configuration\n");
      std::exit(2);
    }
    if (algo.workspaceSize > ws_size) {
      std::fprintf(stderr, "cublasLt workspace %zu exceeds provided %zu\n", algo.workspaceSize,
                   ws_size);
      std::exit(2);
    }
  }

  void run(const void* A, const void* B, float* C, cudaStream_t stream = 0) {
    const float alpha = 1.0f, beta = 0.0f;
    lt_check(cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &algo.algo,
                            workspace, workspace_size, stream),
             "matmul");
  }

  ~LtBatchedFp8() {
    cublasLtMatrixLayoutDestroy(Ad);
    cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
  }
};

// Probe whether cuBLASLt offers an algo for a configuration without
// committing to it (some dtype/layout combos are simply unsupported).
inline bool lt_fp8_supported(cudaDataType_t dt, int n, std::int64_t batch) {
  cublasLtHandle_t lt;
  if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
  bool ok = cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS;
  std::size_t ws = 32u << 20;
  if (ok)
    ok = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                              sizeof(ws)) == CUBLAS_STATUS_SUCCESS;
  auto make_layout = [&](cublasLtMatrixLayout_t* lay, cudaDataType_t t) {
    if (cublasLtMatrixLayoutCreate(lay, t, n, n, n) != CUBLAS_STATUS_SUCCESS) return false;
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    const int batch_i32 = static_cast<int>(batch);
    const std::int64_t stride = static_cast<std::int64_t>(n) * n;
    return cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                            sizeof(order)) == CUBLAS_STATUS_SUCCESS &&
           cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_i32,
                                            sizeof(batch_i32)) == CUBLAS_STATUS_SUCCESS &&
           cublasLtMatrixLayoutSetAttribute(*lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                            &stride, sizeof(stride)) == CUBLAS_STATUS_SUCCESS;
  };
  ok = ok && make_layout(&Ad, dt) && make_layout(&Bd, dt) && make_layout(&Cd, CUDA_R_32F);
  if (ok) {
    cublasLtMatmulHeuristicResult_t hr;
    int count = 0;
    ok = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &hr, &count) ==
             CUBLAS_STATUS_SUCCESS &&
         count > 0;
  }
  cublasLtMatrixLayoutDestroy(Ad);
  cublasLtMatrixLayoutDestroy(Bd);
  cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(desc);
  cublasLtDestroy(lt);
  return ok;
}

}  // namespace ftg

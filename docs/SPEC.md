# SPEC.md — batched tiny FP8 GEMM, spec of record

## Kernel

C = A·B for batches of tiny square matrices. Shapes: M=N=K ∈ {16, 32, 64};
batch ∈ {64, 1024, 65536}. A, B stored quantized (e4m3 default, e5m2 via
flag); C in FP32. One warp per matrix (or sub-warp tile per matrix for
N=16), register-resident: each lane loads its share of A/B rows, converts
FP8→FP32 on the fly (`cuda_fp8.h` device intrinsics, RNE), accumulates the
row·col dot in FP32 registers. No shared-memory staging needed at these
sizes; no tensor-core mma in v1 (CUDA cores — the win being tested is
launch/setup tax, not FLOPs; say so in the benchmark header).

Row-major A, B, C; batch-major (b*M*M + r*M + c), same convention as btl.

## Vendor baseline (the honest configuration)

cuBLASLt batched FP8: `cublasLtMatmul` with CUDA_R_8F_E4M3 inputs,
CUBLAS_COMPUTE_32F, strided-batched descriptor, FP32 output, best
heuristic result (query algo, don't take the default blindly — THAT is
"the library's best honest configuration" per kernel-vs-library's ground
rule).
Warmup + median-of-N timing for BOTH sides; same data.

## Correctness (tests/, ctest)

- FP64 reference GEMM (host, plain loops) on fixed-seed random data.
- Kernel output vs reference: relative L2 error bound derived in the test
  comment from first principles (e4m3: 3 mantissa bits → unit roundoff
  u = 2^-4; K-term dot random walk ~ u·√K·‖a‖‖b‖; assert measured error
  inside 3× that bound, report the actual ratio). e5m2: u = 2^-3, expect
  ~2× worse — assert the ratio lands in [1.5, 3].
- Degenerate inputs: all-zeros, ±max (448 e4m3 / 57344 e5m2) saturation
  behavior REPORTED (no NaN/Inf in output; saturation is counted and
  reported — no silent garbage).

## Benchmark (benchmarks/, JSON out)

- Sweep (size × batch × dtype-kernel-vs-cublaslt), µs per batched call,
  median-of-21 + [min,max], effective TFLOP/s and GB/s.
- Launch-overhead exhibit: time a batch=1 call for both paths (the fixed
  cost the tiny regime amortizes).
- Machine: RTX 5060 Laptop, powersave governor caveat; medians+spreads.
- tools/plot.py: time vs batch per size, both paths, crossover annotated;
  PNG to data/.

## Accuracy appendix (FP4 e2m1)

Same shapes, inputs quantized e2m1 via `cuda_fp4.h`, CUDA-core dot, vs the
same FP64 reference. Accuracy only — no vendor baseline, no performance
claim. Expected: ~2^3 worse than e4m3 (one mantissa bit vs three); the
report states the measured ratio.

## Honesty rules

- Kill criterion: publish the table win or lose; empty win zone is a
  result (btl precedent: cuSOLVER won 2/60 and that table shipped).
- Every kernel launch checked; non-finite outputs counted and reported.
- No CUTLASS, no cuBLAS in the kernel path (baseline only).

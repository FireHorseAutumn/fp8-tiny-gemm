# fp8-tiny-gemm — batched tiny FP8 GEMM vs cuBLASLt (kvl benchmark #6)

Hand-written batched tiny GEMM with FP8 storage and FP32 accumulation,
measured against cuBLASLt's batched FP8 path (the library's best honest
configuration). Registered as benchmark #6 in ../kernel-vs-library.

## Claim

In the launch-tax regime (M=N=K ≤ 64, batch in the thousands+), a
hand-written kernel — one warp per matrix, register-resident tiles, no
workspace query, no per-call setup — beats cuBLASLt batched FP8, for the
same reason batched tiny Cholesky beats cuSOLVER (btl: 58/60, 1.8–160×).
At any real size, cuBLASLt's tensor-core path wins by a wide margin.

## Kill criterion (publish either way)

If cuBLASLt wins at every (size, batch) point, that table ships in the
README with the win zone marked empty. The honest negative is the
artifact's culture.

## Numerics (the point of the exercise)

- Inputs quantized to FP8 e4m3 (default) or e5m2 (flag) with round-to-
  nearest-even via CUDA 13.2 `cuda_fp8.h`; accumulation in FP32
  (CUDA cores for the kernel; tensor cores inside cuBLASLt).
- Accuracy measured against an FP64 reference: error bounds stated from
  first principles (mantissa bits × √K random-walk), not just reported.
- FP4 e2m1 appears as an ACCURACY APPENDIX only (built separately), never
  as the kernel bet.
- See ../gpu-bundle-adjustment's precision study for why FP8 belongs in
  tensor-core-style accumulate paths and never in measurement space.

## Layout

- `docs/SPEC.md` — spec of record (kernel design, baseline config, tests)
- `include/ftg/` — device kernels (header-only, hand-written, no CUTLASS)
- `src/` — host drivers; `benchmarks/` — vs cuBLASLt, JSON out
- `tests/` — correctness vs FP64 oracle, error-bound assertions
- `tools/` — plots; `data/` — results

## Toolchain

CUDA 13.2, sm_120 (RTX 5060 Laptop); g++-15 host compiler (nvcc rejects
g++-16 — auto-detect block in CMakeLists). ninja + ccache.

## Status

Spec only; no implementation or benchmark results yet.

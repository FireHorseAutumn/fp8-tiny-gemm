# fp8-tiny-gemm

Hand-written batched tiny GEMM with FP8 storage and FP32 accumulation,
measured against cuBLASLt's batched FP8 path (the library's best honest
configuration). Registered as benchmark #6 in
[kernel-vs-library](https://github.com/FireHorseAutumn/kernel-vs-library).

## Status

- CTest 2/2 (correctness + FP4 appendix); full benchmark and accuracy
  tables measured and published, win zone n=16.
- Recorded on an RTX 5060 Laptop (sm_120, CUDA 13.2, driver 615.71.09),
  governor-pinned idle machine; µs per batched call, median of 21.

## Headline: the launch-tax win zone is n=16 only

One warp per matrix, register-resident, no workspace query, no per-call
setup, vs cuBLASLt batched FP8. µs per batched call, median of 21 with
[min, max]:

| n | batch | kernel | cuBLASLt | kernel/lt | winner |
|--:|--:|--|--|--|--|
| 16 | 1 | 5.9 [5.2, 13.5] | 6.8 [5.7, 14.0] | 0.86 | kernel |
| 16 | 64 | 5.7 [5.3, 8.9] | 8.8 [7.6, 17.9] | 0.65 | kernel |
| 16 | 1024 | 13.6 [13.2, 14.6] | 17.1 [16.7, 26.4] | 0.79 | kernel |
| 16 | 65536 | 720.6 [707.5, 2282.3] | 828.4 [794.7, 831.4] | 0.87 | kernel |
| 32 | 1 | 20.8 [19.8, 22.5] | 6.8 [6.3, 13.6] | 3.07 | cublaslt |
| 32 | 64 | 20.2 [19.3, 22.0] | 10.8 [8.2, 11.8] | 1.86 | cublaslt |
| 32 | 1024 | 75.7 [74.8, 81.5] | 68.3 [67.9, 77.4] | 1.11 | cublaslt |
| 32 | 65536 | 4874.9 [4249.6, 4921.1] | 4225.4 [4200.3, 4812.5] | 1.15 | cublaslt |
| 64 | 1 | 109.6 [108.0, 110.2] | 6.0 [5.6, 12.0] | 18.12 | cublaslt |
| 64 | 64 | 112.2 [111.4, 121.0] | 11.0 [10.7, 22.1] | 10.22 | cublaslt |
| 64 | 1024 | 604.2 [593.7, 629.9] | 88.5 [87.1, 98.8] | 6.82 | cublaslt |
| 64 | 65536 | 39359.3 [38976.8, 54590.1] | 5726.0 [5550.9, 6912.9] | 6.87 | cublaslt |

## Key numbers

- At n=16 the kernel wins every batch point at 0.65 to 0.87× vendor time;
  at n=32 and n=64 cuBLASLt's tensor-core path wins everywhere, up to
  18.1×. Per the kill criterion the table ships as measured.
- Accuracy vs an FP64 oracle: relative L2 error is 7 to 14% of the
  first-principles bound u·√K; the e5m2/e4m3 ratio is 2.0 as predicted by
  the mantissa count; cuBLASLt matches the kernel's error to printed
  precision.
- Launch-overhead exhibit (batch=1): kernel fixed cost ~5.9 µs vs ~6.8 µs
  for cuBLASLt, the ~1 µs setup tax the tiny regime amortizes.

## Honest limits

- The launch-tax hypothesis survives only at the smallest size tested.
  n=32 and n=64 are vendor wins at every batch, published.
- The n=16 batch=65536 cell is bimodal even governor-pinned (kernel reps
  span 707 to 2282 µs inside a single run; one of four same-morning pinned
  runs lost 1.55× there). The other three n=16 cells won in all four runs.
- e5m2: cuBLASLt's heuristic offers no algo for the row-major strided
  batched configuration on this stack; kernel-side e5m2 tracks e4m3 within
  ~2% (7% at the n=16 batch=1 launch-noise floor).
- FP4 e2m1 exists as an accuracy appendix only: measured 11.1 to 11.3× the
  e4m3 error, above the 4× mantissa-count prediction, because on U(-1,1)
  data the e2m1 grid rounds every input to {0, ±0.5, ±1}. Never the kernel
  bet.

## Build and test

CUDA 13.2, sm_120 (RTX 5060 Laptop); g++-15 host compiler (nvcc rejects
g++-16, auto-detect block in CMakeLists). ninja + ccache.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Repository layout

- `docs/SPEC.md`: spec of record (kernel design, baseline config, tests)
- `include/ftg/`: device kernels and host helpers (header-only, no CUTLASS)
- `benchmarks/`: vs cuBLASLt, JSON out
- `tests/`: correctness vs FP64 oracle, error-bound assertions
- `tools/`: plots; `data/`: results

## Deeper reading

- `docs/RUNGS.md`: the full record: the measured claim with its stability
  caveat, the accuracy study with the FP4 appendix, and the numerics
  rationale.
- `docs/SPEC.md`: the spec of record.
- `data/fp8_bench.json`: raw per-rep timings and machine metadata.

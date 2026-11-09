# Record

The full record for fp8-tiny-gemm (kvl benchmark #6): the measured claim
for batched tiny FP8 GEMM vs cuBLASLt, the pinned result
table, the accuracy study, and the numerics rationale. The README carries
the headline; this file carries the evidence. Recorded on an RTX 5060
Laptop (sm_120, CUDA 13.2, driver 615.71.09) on a governor-pinned idle
machine (performance governor, recorded in `data/fp8_bench.json`); the
table is one pinned run, and the cross-run stability notes below apply.
CTest 2/2 (correctness + FP4 appendix).

## Claim (measured)

In the launch-tax regime a hand-written kernel: one warp per matrix,
register-resident, no workspace query, no per-call setup: was expected to
beat cuBLASLt batched FP8 for M=N=K <= 64. **Measured: the win zone is n=16
only.** At n=16 the kernel wins every batch point in the pinned table
(0.65 to 0.87× vendor time); at n=32 and n=64 cuBLASLt's tensor-core path
wins everywhere, up to 18.1×. The launch-tax hypothesis survives only at
the smallest size tested. Stability caveat: the n=16 batch=65536 cell is
bimodal even governor-pinned (kernel reps span 707 to 2282 µs inside a single
run; one of four same-morning pinned runs lost 1.55× there); the other
three n=16 cells won in all four pinned runs.

## Results

µs per batched call, median of 21 with [min, max]; raw per-rep timings and
machine metadata in `data/fp8_bench.json`.

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

**e5m2:** cuBLASLt's heuristic offers no algo for the row-major strided
batched e5m2 configuration on this stack (recorded as `null` cells in the
JSON); kernel-side e5m2 times are in the sweep and track e4m3 within ~2%
(7% at the n=16 batch=1 launch-noise floor).

Launch-overhead exhibit (batch=1 row): the kernel's fixed cost is ~5.9 µs
vs ~6.8 µs for cuBLASLt: the ~1 µs setup tax the tiny regime amortizes.

![time vs batch](data/fp8_bench.png)

Governor-pinned idle-machine rerun: table re-filled (win zone unchanged;
n=64 batch=1 loss deepened to 18.1x; the n=16 batch=65536 cell is bimodal
even pinned, see the claim caveat). Per the kill criterion the table ships
as measured.

## Accuracy (vs FP64 oracle, batch=64, fixed seeds)

Relative L2 error against an FP64 reference of the unquantized inputs,
compared to the first-principles bound u·√K (u = 2^-4 e4m3, 2^-3 e5m2;
assertion at 3× that):

| n | e4m3 rel | e5m2 rel | e5m2/e4m3 | cuBLASLt e4m3 rel |
|--:|--|--|--|--|
| 16 | 3.39e-2 | 6.74e-2 | 1.99 | 3.39e-2 |
| 32 | 3.34e-2 | 6.71e-2 | 2.01 | 3.34e-2 |
| 64 | 3.34e-2 | 6.69e-2 | 2.00 | 3.34e-2 |

Measured error is 7 to 14% of u·√K (2 to 5% of the 3× assertion bound); the
e5m2/e4m3 ratio is 2.0 as predicted by the mantissa count. cuBLASLt matches
the kernel's error to printed precision: input quantization dominates,
accumulation is FP32 on both sides. FP4 e2m1 appendix (`test_fp4_accuracy`):
measured 11.1 to 11.3× the e4m3 error, well above the 4× mantissa-count
prediction, because on U(-1,1) data the e2m1 grid rounds every input to
{0, ±0.5, ±1}: a quarter of the inputs become exact zeros and the error is
absolute-grid dominated, not relative-roundoff dominated. Degenerate inputs:
all-zeros give exact zeros; ±900-magnitude inputs saturate (SATFINITE
quantization, counted: ~50% of entries) with no NaN/Inf in any output.

## Numerics (the point of the exercise)

- Inputs quantized to FP8 e4m3 (default) or e5m2 (flag) with round-to-
  nearest-even via CUDA 13.2 `cuda_fp8.h`; accumulation in FP32
  (CUDA cores for the kernel; tensor cores inside cuBLASLt).
- Accuracy measured against an FP64 reference: error bounds stated from
  first principles (mantissa bits × √K random-walk), not just reported.
- FP4 e2m1 appears as an ACCURACY APPENDIX only (built separately), never
  as the kernel bet.
- See the [gpu-bundle-adjustment](https://github.com/FireHorseAutumn/gpu-bundle-adjustment) precision study for why FP8 belongs in
  tensor-core-style accumulate paths and never in measurement space.

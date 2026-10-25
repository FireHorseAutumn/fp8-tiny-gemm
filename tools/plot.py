#!/usr/bin/env python3
"""Time vs batch per size, kernel vs cuBLASLt, crossover annotated.
Reads data/fp8_bench.json, writes data/fp8_bench.png."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data/fp8_bench.json") as f:
    bench = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors = {16: "tab:blue", 32: "tab:orange", 64: "tab:green"}
for n in (16, 32, 64):
    rows = [r for r in bench["results"] if r["n"] == n and r["dtype"] == "e4m3"]
    batches = [r["batch"] for r in rows]
    ax.plot(batches, [r["kernel_us"] for r in rows], "o-", color=colors[n],
            label=f"kernel n={n}")
    vendor_rows = [r for r in rows if r.get("cublaslt_us") is not None]
    ax.plot([r["batch"] for r in vendor_rows], [r["cublaslt_us"] for r in vendor_rows], "s--",
            color=colors[n], alpha=0.65, label=f"cuBLASLt n={n}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("batch")
ax.set_ylabel("us per batched call (median of 21)")
ax.set_title(f"fp8 tiny GEMM vs cuBLASLt, e4m3 ({bench['device']})", fontsize=10)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
ax.annotate("win zone: n=16 only", xy=(1e3, 16), xytext=(30, 200),
            arrowprops=dict(arrowstyle="->", alpha=0.6), fontsize=9)
fig.tight_layout()
fig.savefig("data/fp8_bench.png", dpi=150)
print("wrote data/fp8_bench.png")

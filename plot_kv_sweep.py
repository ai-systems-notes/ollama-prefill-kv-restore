#!/usr/bin/env python3
"""Plot the prefix-length sweep: TTFT scaling (recompute vs KV restore) + speedup."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/bench_ollama_kv_sweep.json") as fh:
    data = json.load(fh)
rows = data["rows"]
x   = [r["n_saved"] for r in rows]
nr  = [r["norestore_ms"] for r in rows]
rs  = [r["restore_ms"] for r in rows]
sp  = [r["speedup"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# --- left: TTFT, log-log ---
ax1.plot(x, nr, "o-", color="#d1495b", lw=2, ms=6, label="Recompute (no restore)")
ax1.plot(x, rs, "s-", color="#2e8b57", lw=2, ms=6, label="KV restore")
ax1.set_xscale("log", base=2)
ax1.set_yscale("log")
ax1.set_xlabel("System-prompt length (tokens)")
ax1.set_ylabel("TTFT — prefill time (ms)")
ax1.set_title("TTFT: recompute grows linearly, restore stays flat")
ax1.grid(True, which="both", ls=":", alpha=0.5)
ax1.legend(loc="upper left", frameon=False)
ax1.annotate(f"{nr[-1]:.0f} ms", (x[-1], nr[-1]), textcoords="offset points",
             xytext=(-10, 8), ha="right", color="#d1495b", fontsize=9)
ax1.annotate(f"{rs[-1]:.0f} ms", (x[-1], rs[-1]), textcoords="offset points",
             xytext=(-10, -14), ha="right", color="#2e8b57", fontsize=9)

# --- right: speedup ---
ax2.plot(x, sp, "D-", color="#3a6ea5", lw=2, ms=6)
ax2.set_xscale("log", base=2)
ax2.set_xlabel("System-prompt length (tokens)")
ax2.set_ylabel("TTFT speedup (×)")
ax2.set_title("Speedup grows with prefix length")
ax2.grid(True, which="both", ls=":", alpha=0.5)
for xi, si in zip(x, sp):
    ax2.annotate(f"{si:.0f}×", (xi, si), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=8, color="#3a6ea5")

fig.suptitle(f"Ollama KV slot save/restore — {data['model']}, RTX 4070 "
             f"(mean of {data['repeat']}, cold start excluded)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/kv_sweep.png", dpi=150)
print("wrote results/kv_sweep.png")

# ollama-prefill-kv-restore

**Opt-in KV-cache save/restore for Ollama — up to 417× lower reported prefill-processing time versus recomputation of a fixed system prompt after cache eviction.**

A single-user local agent usually prepends the *same* large system prompt
(rules, tool definitions) on every turn. When that prefix is no longer available
in the in-memory cache, it must be recomputed. This benchmark compares that
recomputation with reuse of a previously saved KV state.

The bundled `llama-server` already knows how to persist a slot's KV cache to disk
(`--slot-save-path`), but Ollama never passes the flag and never exposes the
endpoints. This work adds a small **opt-in** path so that the fixed prefix can be
prefilled *once*, saved, and **restored** before later requests. In this benchmark,
the reported prefill-processing time after restoration is roughly 9–19 ms.

This repo holds the **reproducible benchmark and results**. The implementation is
proposed upstream to Ollama (see [Upstream](#upstream)).

## Headline result

`llama3.2:3b`, RTX 4070, flash attention on, single-user serial requests.
The measured quantity is the prefill-processing time reported by llama.cpp
`/completion`, averaged over 3 runs after forced cache eviction, with one warm-up
iteration discarded. The maximum 417× ratio compares recomputation of a fixed
33,486-token system prefix with the reported prefill-processing time after KV
restoration. It is not a measurement of client-observed end-to-end TTFT and does
not include the separate restore API latency, model loading, or network time.

| System prompt (tok) | Recompute prefill time | Prefill time after restore | Ratio |
|---:|---:|---:|---:|
| 264 | 30.2 ms | 9.1 ms | 3.3× |
| 516 | 53.9 ms | 8.9 ms | 6.0× |
| 1,020 | 108.5 ms | 9.1 ms | 11.9× |
| 4,107 | 476.7 ms | 10.4 ms | 45.9× |
| 8,223 | 1,067 ms | 11.4 ms | 93.6× |
| 16,455 | 2,692 ms | 13.9 ms | 193× |
| 33,486 | 8,110 ms | 19.4 ms | **417×** |

![Reported prefill-processing time and recompute/restore ratio](results/kv_sweep.png)

**Reported prefill time after restore is roughly flat (~9–19 ms); recomputation
grows approximately with prefix length**. The measured ratio increases from
3.3× at 264 tokens to **417× at 33,486 tokens** under these conditions.
These figures concern the prefill metric only; they do not establish the same
ratio for end-to-end TTFT. Decode throughput (tokens/s) is unchanged.

## How it works

When the opt-in env var `OLLAMA_SLOT_SAVE_PATH` is set, the forked Ollama:

1. launches `llama-server` with `--slot-save-path <dir>`, and
2. exposes two experimental endpoints that go through the runner (serialized
   against inference via the same semaphore as the completion path):
   - `POST /api/experimental/kv/save`    `{"model": "...", "filename": "sys.bin"}`
   - `POST /api/experimental/kv/restore` `{"model": "...", "filename": "sys.bin"}`

When the env var is **unset**, behaviour is byte-for-byte identical to stock
Ollama. Save/restore requires `OLLAMA_NUM_PARALLEL=1` (single-user, serial-request
use case); `filename` is restricted to a basename so saves can't escape the dir.

## Reproduce

Requires the forked Ollama running with the feature enabled (single user):

```bash
OLLAMA_SLOT_SAVE_PATH=/tmp/ollama-kv \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_FLASH_ATTENTION=true \
OLLAMA_CONTEXT_LENGTH=34816 \
  ollama serve            # on a non-default port, e.g. :11436
```

Then:

```bash
pip install matplotlib            # only plot_kv_sweep.py needs it
python bench_ollama_kv_sweep.py   # writes results/bench_ollama_kv_sweep.json
python plot_kv_sweep.py           # writes results/kv_sweep.png
```

The benchmark sweeps system-prompt length 256 → ~33K tokens. For each length it
warms and **saves** the system prefix, then per iteration: evicts slot 0 and
measures the reported **recompute prefill time**, evicts again, **restores** the
saved KV, and measures the reported **prefill time after restore**. One warm-up iteration is discarded; three are
averaged. Points are labelled by the exact saved token count (`n_saved`).

`OLLAMA` (host:port) and `MODEL` are overridable via env vars at the top of
`bench_ollama_kv_sweep.py`.

## Files

- [`bench_ollama_kv_sweep.py`](bench_ollama_kv_sweep.py) — the prefix-length sweep.
- [`plot_kv_sweep.py`](plot_kv_sweep.py) — renders the two-panel figure.
- [`results/bench_ollama_kv_sweep.json`](results/bench_ollama_kv_sweep.json) — raw timings (per-run, not just means).
- [`results/kv_sweep.png`](results/kv_sweep.png) — TTFT scaling (left) + speedup (right).

## Notes

- A model whose GGUF embeds a projector (e.g. some VL models) makes Ollama launch
  `llama-server` with `--mmproj` → multimodal mode, which rejects slot
  save/restore with `501 "not supported by multimodal"`. Use a pure text model
  (`llama3.2:3b`, `qwen2.5:3b`, `gemma2:2b`).
- This targets the single-user, serial-request path — not the multi-tenant path.

## Upstream

The implementation is proposed to the Ollama project as an opt-in,
off-by-default feature (`OLLAMA_SLOT_SAVE_PATH`, mounted under
`/api/experimental/`). This repo is the reproducibility companion: scripts, raw
JSON, and the plot.

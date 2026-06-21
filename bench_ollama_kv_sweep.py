#!/usr/bin/env python3
"""Prefix-length sweep for the forked Ollama's /api/experimental/kv/{save,restore}.

Measures, per system-prompt length, how TTFT scales:
  * NO_RESTORE : evict slot 0 with an unrelated prefix, then re-send the system
                 prompt with cache_prompt=true -> the system prefill is RECOMPUTED.
  * RESTORE    : same eviction, then /api/experimental/kv/restore, then re-send
                 -> only the short question tail is computed.

Methodology (matches the requested protocol):
  * pure TEXT model (llama3.2:3b) so save/restore works with OLLAMA_SLOT_SAVE_PATH
    alone -- no text-only flag, reproducible against the upstream PR.
  * cold start excluded: 1 warm-up iteration per length is discarded.
  * each measured point = mean of 3 iterations.
  * TTFT read from llama.cpp /completion `timings` (prompt_ms), the true prefill
    cost; prompt_n = tokens computed, cache_n = tokens reused.
  * lengths labelled by the EXACT saved token count (n_saved), not the target.

Run with the fork on :11436 (OLLAMA_SLOT_SAVE_PATH set, NUM_PARALLEL=1,
OLLAMA_CONTEXT_LENGTH high enough for the largest prompt + tail).
"""
import json, os, subprocess, urllib.request, statistics as st

OLLAMA = "http://127.0.0.1:11436"
MODEL = os.environ.get("MODEL", "llama3.2:3b")
TARGETS = [256, 512, 1024, 4096, 8192, 16384, 32768]
WARMUP = 1     # discarded cold-start iterations
REPEAT = 3     # measured iterations, averaged

def post(url, p, timeout=600):
    r = urllib.request.Request(url, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())

HEADER = ("You are an expert software engineering assistant operating inside a CLI.\n"
          "Follow these rules at all times:\n")
FOOTER = "When answering arithmetic, output ONLY the final number with no explanation.\n"
RULE = "- Rule {i}: always be precise, cite file paths, and avoid speculation when facts are available.\n"

def system_of(target_tok):
    # measured ~20.9 tok/rule line + ~33 tok header/footer; pick the line count
    # so the ACTUAL saved token count lands near the target (exact value is
    # reported from n_saved). Largest target must leave room for the tail under
    # the server context (34816).
    n = max(1, round((target_tok - 33) / 20.9))
    return HEADER + "".join(RULE.format(i=i) for i in range(1, n + 1)) + FOOTER

# unrelated prefix used to evict slot 0 (only needs a different first token; kept
# short so eviction itself is cheap and never dominates the measurement)
EVICT = ("Totally unrelated preamble for cache eviction.\n"
         + "".join(f"- Note {i}: lorem ipsum dolor sit amet consectetur adipiscing.\n" for i in range(1, 60)))

def subprocess_port():
    out = subprocess.check_output(["bash", "-lc",
        "pgrep -af llama-server | grep -v bash | grep -o 'port [0-9]*' | awk '{print $2}' | head -1"]).decode().strip()
    return int(out)

def main():
    # load the model under the fork at the server's configured context length
    post(OLLAMA + "/api/generate",
         {"model": MODEL, "prompt": "warm", "raw": True, "stream": False,
          "options": {"num_predict": 1, "temperature": 0}})
    P = subprocess_port()
    SUB = f"http://127.0.0.1:{P}/completion"
    print(f"subprocess llama-server port = {P}\n")

    def comp(prompt, cache):
        t = post(SUB, {"prompt": prompt, "n_predict": 4, "temperature": 0,
                       "seed": 0, "cache_prompt": cache, "id_slot": 0})["timings"]
        return {"prompt_n": t["prompt_n"], "cache_n": t["cache_n"], "prompt_ms": t["prompt_ms"]}

    rows = []
    print(f"{'target':>7} {'n_saved':>8} {'norestore_ms':>13} {'restore_ms':>11} {'speedup':>8} {'reuse_tok':>10}")
    for target in TARGETS:
        system = system_of(target)
        tail = system + "User: Compute 17+24.\nAssistant:"

        # warm the system prefix alone, then SAVE via the Ollama fork
        warm = comp(system, True)
        n_saved = warm["prompt_n"] + warm["cache_n"]
        fn = f"sweep_{target}.bin"
        save = post(OLLAMA + "/api/experimental/kv/save", {"model": MODEL, "filename": fn})

        norest, rest = [], []
        for it in range(WARMUP + REPEAT):
            # NO_RESTORE: evict, then recompute the system prefill
            comp(EVICT, False)
            nr = comp(tail, True)
            # RESTORE: evict again, restore the saved KV, then re-send
            comp(EVICT, False)
            post(OLLAMA + "/api/experimental/kv/restore", {"model": MODEL, "filename": fn})
            rs = comp(tail, True)
            if it >= WARMUP:          # discard cold-start warm-up
                norest.append(nr)
                rest.append(rs)

        nr_ms = st.mean(r["prompt_ms"] for r in norest)
        rs_ms = st.mean(r["prompt_ms"] for r in rest)
        reuse = st.mean(r["cache_n"] for r in rest)
        sp = nr_ms / rs_ms if rs_ms else 0
        rows.append({"target": target, "n_saved": n_saved,
                     "save_ms": save.get("timings", {}).get("save_ms"),
                     "file_bytes": save.get("n_written"),
                     "norestore_ms": nr_ms, "restore_ms": rs_ms,
                     "speedup": sp, "reuse_tok": reuse,
                     "norestore_runs": norest, "restore_runs": rest})
        print(f"{target:>7} {n_saved:>8} {nr_ms:>13.1f} {rs_ms:>11.1f} {sp:>7.1f}x {reuse:>10.0f}")

    os.makedirs("results", exist_ok=True)
    with open("results/bench_ollama_kv_sweep.json", "w") as fh:
        json.dump({"model": MODEL, "warmup": WARMUP, "repeat": REPEAT, "rows": rows}, fh, indent=2)
    print("\nwrote results/bench_ollama_kv_sweep.json")

if __name__ == "__main__":
    main()

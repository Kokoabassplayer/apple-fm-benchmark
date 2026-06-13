"""Orchestrator: run GSM8K, HumanEval, MMLU and aggregate everything into
results/all.json. Embeds the already-measured spec/latency/context/privacy JSONs
(they are NOT re-run). Each suite writes its own JSON as soon as it finishes so
partial progress survives a crash.
"""
from __future__ import annotations
import sys, os, json, time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "results")
sys.path.insert(0, HERE)

import fm_common as F  # noqa: E402
import gsm8k, humaneval, mmlu  # noqa: E402


def load_existing(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing {p} — must exist before run_all")
    with open(p) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    total_t0 = time.monotonic()

    print("=== GSM8K ===", flush=True)
    t = time.monotonic()
    g = gsm8k.run(n=gsm8k.N)
    F.write_json(os.path.join(RESULTS, "gsm8k.json"), g)
    print(f"GSM8K: {g['correct']}/{g['n']} = {g['accuracy']}  ({time.monotonic()-t:.0f}s)", flush=True)

    print("=== HumanEval ===", flush=True)
    t = time.monotonic()
    h = humaneval.run()
    F.write_json(os.path.join(RESULTS, "humaneval.json"), h)
    print(f"HumanEval: {h['passed']}/{h['n']} pass@1={h['pass_at_1']}  ({time.monotonic()-t:.0f}s)", flush=True)

    print("=== MMLU ===", flush=True)
    t = time.monotonic()
    m = mmlu.run()
    F.write_json(os.path.join(RESULTS, "mmlu.json"), m)
    print(f"MMLU: {m['correct']}/{m['n']} = {m['accuracy']}  ({time.monotonic()-t:.0f}s)", flush=True)

    total_runtime = time.monotonic() - total_t0

    spec = load_existing("spec.json")
    latency = load_existing("latency.json")
    context = load_existing("context.json")
    privacy = load_existing("privacy.json")

    all_out = {
        "spec": spec,
        "latency": latency,
        "context": context,
        "privacy": privacy,
        "suites": {
            "gsm8k": g,
            "humaneval": h,
            "mmlu": m,
        },
        "total_runtime_s": round(total_runtime, 1),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    F.write_json(os.path.join(RESULTS, "all.json"), all_out)

    print("\n=== SUMMARY ===", flush=True)
    print(f"GSM8K     accuracy = {g['accuracy']}  ({g['correct']}/{g['n']})", flush=True)
    print(f"HumanEval pass@1   = {h['pass_at_1']}  ({h['passed']}/{h['n']})", flush=True)
    print(f"MMLU      accuracy = {m['accuracy']}  ({m['correct']}/{m['n']})", flush=True)
    print(f"Total runtime: {total_runtime:.0f}s", flush=True)


if __name__ == "__main__":
    main()

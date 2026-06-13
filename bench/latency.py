#!/usr/bin/env python3
"""Rigorous latency measurement for the on-device `fm` model.

Four things, in the order that makes them honest:

  1. COLD ttft  — first streamed call in the process (model not warm).
     Captured before any warmup.
  2. warmup     — discarded calls so the runtime is resident.
  3. prefill    — stream a trivial reply across input sizes
     {256,512,1024,2048,3072} (capped under the ~4k context ceiling); record
     time-to-first-token. For SHORT outputs streaming ttft is honest, and the
     SLOPE of ttft vs input length is the prefill rate (tokens/s); intercept
     absorbs process spawn + 1-token decode. Warm-TTFT stats come from the
     smallest input size here (short output => reliable incremental flush).
  4. decode     — `fm --stream` BUFFERS long outputs and flushes at completion,
     so ttft is useless for long generations (ttft ~= wall). We therefore use
     `--no-stream` and fit total WALL TIME vs OUTPUT LENGTH across counts of
     {120,250,450}. Slope = 1/decode_rate; intercept = spawn+prefill. Only the
     reliable total wall time enters the fit.

Outputs results/latency.json. Pure stdlib (statistics.linear_regression).
"""
import statistics, time
import fm_common as F

SIZES = [256, 512, 1024, 2048, 3072]     # capped: 4096 input exceeds ~4k ceiling
PREFILL_N = 4
DECODE_LENS = [120, 250, 450]            # count-to-N targets (vary output length)
DECODE_N = 4                             # repeats per length
WARMUP = 3

TRIVIAL = "\n\nReply with the single word OK and nothing else."


def now(): return time.monotonic()


def prefill_run(size):
    filler = F.filler_of(size)
    actual_in = F.tok_count(filler + TRIVIAL)
    r = F.respond_stream(filler + TRIVIAL)
    return {"size_target": size, "actual_input_tokens": actual_in,
            "ttft": r["ttft"], "wall": r["wall_s"], "rc": r["returncode"]}


def decode_run(target_n):
    prompt = (f"Count from 1 to {target_n}. Output the numbers separated by "
              "single spaces. No words, no punctuation, no commentary.")
    r = F.respond(prompt)                 # --no-stream: total wall is honest
    out_tok = F.tok_count(r["text"])
    nums = len([w for w in r["text"].replace(",", " ").split() if w.strip().isdigit()])
    return {"target_n": target_n, "wall": r["wall_s"], "out_tokens": out_tok,
            "numbers_emitted": nums, "rc": r["returncode"],
            "sample": r["text"][:60]}


def linfit(xs, ys):
    if len(xs) >= 2:
        slope, intercept = statistics.linear_regression(xs, ys)
        return slope, intercept, (1.0 / slope) if slope else None
    return None, None, None


def main():
    out = {"_unit_notes": {
        "prefill_tps": "1/slope of (ttft vs input tokens); honest prefill speed",
        "decode_tps": "1/slope of (wall vs output tokens); honest decode speed",
        "warm_ttft_s": "median ttft on smallest (256-tok) input; short output flushes promptly",
    }}

    # 1. COLD (first thing in the process)
    print("[1/4] cold ttft ...")
    cold = F.respond_stream("Reply with OK." + TRIVIAL)
    out["cold"] = {"ttft": cold["ttft"], "wall": cold["wall_s"], "rc": cold["returncode"]}

    # 2. warmup
    print(f"[2/4] warmup x{WARMUP} ...")
    for _ in range(WARMUP):
        F.respond_stream("Reply with OK." + TRIVIAL)

    # 3. prefill sweep
    print("[3/4] prefill sweep", SIZES, f"x{PREFILL_N}")
    per_size = {}
    for s in SIZES:
        runs = [prefill_run(s) for _ in range(PREFILL_N)]
        tts = [x["ttft"] for x in runs if x["ttft"]]
        med = statistics.median(tts) if tts else None
        med_in = statistics.median([x["actual_input_tokens"] for x in runs if x["actual_input_tokens"]])
        per_size[s] = {"median_ttft": round(med, 4) if med else None,
                       "median_input_tokens": med_in, "runs": runs}
        print(f"   {s:>5}tok in -> ttft {med:.3f}s" if med else f"   {s:>5}tok FAILED")
        time.sleep(0.15)

    xs, ys = [], []
    for s in SIZES:
        d = per_size[s]
        if d["median_ttft"] and d["median_input_tokens"]:
            xs.append(d["median_input_tokens"]); ys.append(d["median_ttft"])
    slope, intercept, prefill_rate = linfit(xs, ys)
    out["prefill"] = {
        "input_sizes": SIZES, "per_size": per_size,
        "fit": {"slope_s_per_tok": slope, "intercept_s": intercept,
                "prefill_rate_tps": prefill_rate,
                "model": "ttft_s = intercept + slope * input_tokens"},
    }
    # warm ttft stats from the smallest input (short output => honest incremental)
    warm_ttfts = [x["ttft"] for x in per_size[SIZES[0]]["runs"] if x["ttft"]]
    out["warm_ttft"] = F.stats(warm_ttfts)
    if prefill_rate:
        print(f"   PREFILL fit: {prefill_rate:.0f} tok/s "
              f"(ttft = {intercept:.3f} + {slope:.5f}*n)")
    print(f"   warm ttft p50: {out['warm_ttft']['median']}s")

    # 4. decode (wall vs output length)
    print("[4/4] decode wall-vs-length", DECODE_LENS, f"x{DECODE_N}")
    runs = []
    for n in DECODE_LENS:
        for _ in range(DECODE_N):
            runs.append(decode_run(n))
            time.sleep(0.1)
    dxs, dys = [], []
    per_len = {}
    for n in DECODE_LENS:
        sub = [r for r in runs if r["target_n"] == n]
        walls = [r["wall"] for r in sub if r["wall"]]
        toks = [r["out_tokens"] for r in sub if r["out_tokens"]]
        per_len[n] = {"median_wall": round(statistics.median(walls), 3) if walls else None,
                      "median_out_tokens": round(statistics.median(toks)) if toks else None,
                      "median_tps": (statistics.median(toks) / statistics.median(walls))
                                    if walls and toks else None}
        print(f"   count->{n}: wall {per_len[n]['median_wall']}s, "
              f"{per_len[n]['median_out_tokens']} tok, "
              f"{per_len[n]['median_tps']:.1f} tok/s" if per_len[n]['median_tps'] else f"   count->{n}: FAILED")
        dxs += [r["out_tokens"] for r in sub if r["out_tokens"]]
        dys += [r["wall"] for r in sub if r["wall"]]
    dslope, dintercept, decode_rate = linfit(dxs, dys)
    out["decode"] = {
        "lengths": DECODE_LENS, "per_length": per_len, "runs": runs,
        "fit": {"slope_s_per_tok": dslope, "intercept_s": dintercept,
                "decode_rate_tps": decode_rate,
                "model": "wall_s = intercept + slope * output_tokens"},
    }
    if decode_rate:
        print(f"   DECODE fit: {decode_rate:.1f} tok/s "
              f"(wall = {dintercept:.3f} + {dslope:.5f}*out_tok)")

    out["methodology"] = (
        f"Cold = first streamed call before warmup. Prefill = median ttft over "
        f"{PREFILL_N} runs per input size {SIZES}, fit linearly (ttft vs input). "
        f"Decode = total wall over {DECODE_N} runs x counts {DECODE_LENS}, fit "
        f"linearly (wall vs output tokens) — no-stream because long outputs "
        f"flush at completion, making ttft~=wall. Warm ttft from {SIZES[0]}-tok "
        f"runs (short output). Greedy throughout (--greedy) for reproducibility."
    )
    F.write_json("results/latency.json", out)
    return out


if __name__ == "__main__":
    main()

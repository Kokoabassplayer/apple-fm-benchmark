#!/usr/bin/env python3
"""Context-window measurement, in two parts:

  1. FUNCTIONAL CEILING (binary search) — the largest input at which the model
     still correctly retrieves a needle placed at the very end. Reuses the
     guardrail-safe Brindleford essay + "chapel wall year 1482" needle.
     Distinguishes a refusal (guardrail) from a true context-size error
     ("exceeded the model's context size"). This is the "fits" number.

  2. MULTI-NEEDLE RECALL CURVE — at several total lengths, embed 3 distinct
     facts at front / middle / back positions, then ask each separately.
     Reports recall fraction vs length & position. This is the "understands"
     number: a model can FIT 3000 tokens yet FAIL to recall a fact buried in
     the middle, so the ceiling alone overstates usable context.

Outputs results/context.json. Stdlib only.
"""
import statistics, time
import fm_common as F

NEEDLE = (" On the wall of the old stone chapel at the edge of town, a date is "
          "carved into the lintel: the year 1482.\n\n"
          "Question: What year is carved into the chapel wall? "
          "Answer with just the year.")
REFUSE = ("unable to work", "i can't", "i cannot", "i'm not able",
          "cannot help", "content policy", "not appropriate", "i won't")
CTX_ERR_MARKERS = ("context size", "context length", "exceeded")


def probe_ceiling(target_tokens):
    filler = F.filler_of(target_tokens) + NEEDLE
    actual = F.tok_count(filler)
    r = F.respond(filler, greedy=True)
    out = r["text"]; low = out.lower()
    ok = "1482" in out
    refused = (not ok) and any(x in low for x in REFUSE)
    ctx_err = r["returncode"] != 0 or any(x in low for x in CTX_ERR_MARKERS)
    return {"target": target_tokens, "actual": actual, "ok": ok,
            "refused": refused, "ctx_err": ctx_err, "rc": r["returncode"],
            "wall": round(r["wall_s"], 2), "out": out[:40], "err": r["stderr"]}


def ceiling():
    print("[context] functional ceiling (binary search)")
    sizes = [500, 1000, 2000, 3000, 4000]
    runs, last_ok, first_fail = [], None, None
    for s in sizes:
        r = probe_ceiling(s)
        runs.append(r); print("  ", {k: r[k] for k in ("target", "actual", "ok", "ctx_err")})
        if r["ok"]:
            last_ok = s
        elif r["refused"]:
            if last_ok is None:
                last_ok = s
        else:
            first_fail = s; break
    if last_ok and first_fail and (first_fail - last_ok) > 500:
        lo, hi = last_ok, first_fail
        while (hi - lo) > 500:
            mid = (lo + hi) // 2
            r = probe_ceiling(mid); runs.append(r)
            print("  BS", {k: r[k] for k in ("target", "actual", "ok", "ctx_err")})
            if r["ok"] or r["refused"]:
                lo = mid
            else:
                hi = mid
        last_ok = lo
    return {"functional_ceiling_tokens": last_ok, "runs": runs}


# 3 distinct, unambiguous facts. Each inserted at a fractional position.
RECALL_NEEDLES = [
    ("harbor", "the harbor master is named Caldwell",
     "What is the name of the harbor master? Answer with just the name."),
    ("tax", "the town sets its market tax rate at 7 percent",
     "What is the market tax rate, in percent? Answer with just the number."),
    ("festival", "the autumn festival always begins on the 19th",
     "On what day does the autumn festival begin? Answer with just the number."),
]
RECALL_EXPECTED = {"harbor": "caldwell", "tax": "7", "festival": "19"}
LENGTHS = [512, 1024, 2048, 3000]


def recall_at_length(total_tokens):
    """Place 3 needles at ~15%/50%/85% of a doc of ~total_tokens; query each."""
    base = F.filler_of(total_tokens)
    # split into 3 segments and drop a needle at each junction
    a = int(len(base) * 0.30); b = int(len(base) * 0.60)
    doc = (base[:a] + "\n\n" + RECALL_NEEDLES[0][1] + ".\n\n" +
           base[a:b] + "\n\n" + RECALL_NEEDLES[1][1] + ".\n\n" +
           base[b:] + "\n\n" + RECALL_NEEDLES[2][1] + ".\n")
    actual = F.tok_count(doc)
    results = []
    for key, _, question in RECALL_NEEDLES:
        r = F.respond(doc + "\n\n" + question, greedy=True)
        ans = r["text"].strip().lower()
        ok = RECALL_EXPECTED[key] in ans
        results.append({"key": key, "question": question.strip(),
                        "answer": r["text"][:40], "ok": ok})
        time.sleep(0.1)
    recalled = sum(1 for x in results if x["ok"])
    return {"total_tokens": actual, "recalled": recalled, "of": len(results),
            "recall_fraction": round(recalled / len(results), 3),
            "needles": results}


def recall_curve():
    print("[context] multi-needle recall curve", LENGTHS)
    rows = []
    for t in LENGTHS:
        row = recall_at_length(t)
        rows.append(row)
        print(f"   {t:>5}tok: recall {row['recalled']}/{row['of']}  "
              f"({row['recall_fraction']:.0%})")
    return {"lengths": LENGTHS, "curve": rows}


def main():
    out = {"ceiling": ceiling(), "recall_curve": recall_curve()}
    out["methodology"] = (
        "Ceiling = largest input where the model retrieves a final needle "
        "(binary search, guardrail-safe filler, refusal vs context-error "
        "distinguished). Recall = 3 facts at 30/60/90% of docs of "
        f"{LENGTHS} tokens, queried separately; recall_fraction = correct/3. "
        "Greedy. Ceiling=can-FIT; recall=can-RETRIEVE."
    )
    F.write_json("results/context.json", out)


if __name__ == "__main__":
    main()

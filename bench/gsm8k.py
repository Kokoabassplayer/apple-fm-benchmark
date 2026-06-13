"""GSM8K accuracy benchmark for Apple Foundation Models (`fm` CLI).

Deterministic: first 300 problems in file order (no seed needed). Greedy decoding.
Prompt asks for step-by-step reasoning ending in `Answer: N`. The scorer takes the
LAST integer in the response and compares it to the ground-truth integer (the number
following `#### ` in the dataset answer).
"""
from __future__ import annotations
import sys, os, re, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_common as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gsm8k_test.jsonl")
RESULTS_DIR = os.path.join(HERE, "..", "results")
N = 300

_GT_INT = re.compile(r"####\s*(-?\d+)")
_LAST_INT = re.compile(r"-?\d[\d,]*")


def load(n=None):
    rows = []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows[:n] if n else rows


def ground_truth(answer_text: str):
    m = _GT_INT.search(answer_text)
    if not m:
        return None
    return int(m.group(1))


def last_integer(text: str):
    """Return the last integer-looking token in text (commas stripped), or None."""
    if not text:
        return None
    matches = list(_LAST_INT.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(0).replace(",", ""))


def build_prompt(question: str) -> str:
    return (
        question.strip()
        + "\n\nSolve step by step. End with 'Answer: N' where N is the final integer."
    )


def run(n=N, verbose=False, progress_every=25):
    rows = load(n)
    samples = []
    correct = 0
    t0 = time.monotonic()
    for i, r in enumerate(rows):
        expected = ground_truth(r["answer"])
        prompt = build_prompt(r["question"])
        try:
            res = F.respond(prompt, greedy=True)
            text = res["text"]
            rc = res["returncode"]
        except Exception as e:
            text, rc = "", -99
            print(f"  [warn] fm call {i} raised: {e}", file=sys.stderr)
        got = last_integer(text)
        is_correct = (expected is not None) and (got is not None) and (got == expected)
        if is_correct:
            correct += 1
        samples.append({
            "i": i,
            "expected": expected,
            "got": got if got is not None else text[-40:],
            "correct": bool(is_correct),
        })
        if verbose:
            print(f"  [{i}] expected={expected} got={got} correct={is_correct}"
                  + ("" if rc == 0 else f" (rc={rc})"))
        if (i + 1) % progress_every == 0:
            acc = correct / (i + 1)
            el = time.monotonic() - t0
            print(f"  GSM8K {i+1}/{len(rows)}  acc={acc:.3f}  elapsed={el:.0f}s", flush=True)
    runtime_s = time.monotonic() - t0
    n_actual = len(rows)
    accuracy = correct / n_actual if n_actual else 0.0
    out = {
        "suite": "GSM8K",
        "source": "openai/grade-school-math (test)",
        "license": "MIT",
        "n": n_actual,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "seed": "deterministic file-order first-300",
        "runtime_s": round(runtime_s, 1),
        "samples": samples,
    }
    return out


def main():
    # CLI flag: `python gsm8k.py spot` runs first 5 verbosely (no checkpoint write)
    if len(sys.argv) > 1 and sys.argv[1] == "spot":
        out = run(n=5, verbose=True)
        print(json.dumps({k: out[k] for k in ("n", "correct", "accuracy")}, indent=2))
        return
    out = run(n=N)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    F.write_json(os.path.join(RESULTS_DIR, "gsm8k.json"), out)
    print(f"GSM8K DONE: {out['correct']}/{out['n']} = {out['accuracy']:.4f} "
          f"in {out['runtime_s']}s")


if __name__ == "__main__":
    main()

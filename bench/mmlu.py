"""MMLU accuracy benchmark for Apple Foundation Models (`fm` CLI).

Stratified ~500 questions across the 57 subjects: group the CSV by subject and
deterministically stride-sample ~9 per subject. Greedy. The prompt shows the
question with choices labeled A-D and asks for just the letter. The scorer takes
the first A/B/C/D in the response (uppercased) and compares to ground truth.
"""
from __future__ import annotations
import sys, os, re, csv, time, json
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_common as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "mmlu_test.csv")
RESULTS_DIR = os.path.join(HERE, "..", "results")
TARGET_PER_SUBJECT = 9   # 57 subjects * ~9 ≈ 500+
_LETTER = re.compile(r"[ABCD]")


def load_all():
    with open(DATA, newline="") as f:
        return list(csv.DictReader(f))


def stratified_sample(rows, per_subject=TARGET_PER_SUBJECT):
    """Deterministic round-robin: within each subject take items at a fixed stride
    so we get ~per_subject evenly across the subject's rows. Stable order."""
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(r)
    # deterministic subject order
    subjects = sorted(by_subject.keys())
    out = []
    for s in subjects:
        items = by_subject[s]
        n = len(items)
        # stride so we pick ~per_subject evenly; if subject has fewer, take all
        k = min(per_subject, n)
        idxs = [int(round((i + 1) * n / (k + 1))) - 1 for i in range(k)]
        # dedupe + clamp (round() can collide on tiny subjects)
        seen = set()
        for idx in idxs:
            idx = max(0, min(n - 1, idx))
            if idx not in seen:
                seen.add(idx)
                out.append(items[idx])
        # if collisions dropped some, top up from the front
        if len(seen) < k:
            for idx in range(n):
                if len(seen) >= k:
                    break
                if idx not in seen:
                    seen.add(idx)
                    out.append(items[idx])
    return out, subjects


def build_prompt(q):
    return (
        q["question"].strip()
        + "\n\nA. " + q["A"]
        + "\nB. " + q["B"]
        + "\nC. " + q["C"]
        + "\nD. " + q["D"]
        + "\n\nAnswer with just the letter (A, B, C, or D)."
    )


def parse_letter(text):
    if not text:
        return None
    m = _LETTER.search(text.upper())
    return m.group(0) if m else None


def run(verbose=False, progress_every=25):
    rows = load_all()
    sample, subjects = stratified_sample(rows)
    samples = []
    by_subject = OrderedDict((s, {"n": 0, "correct": 0}) for s in subjects)
    correct = 0
    t0 = time.monotonic()
    for i, q in enumerate(sample):
        expected = q["answer"].strip().upper()
        if expected not in ("A", "B", "C", "D"):
            # malformed ground truth -> skip scoring but count the row
            samples.append({"i": i, "subject": q["subject"], "expected": expected,
                            "got": None, "correct": False})
            by_subject[q["subject"]]["n"] += 1
            continue
        try:
            res = F.respond(build_prompt(q), greedy=True)
            text = res["text"]
            rc = res["returncode"]
        except Exception as e:
            text, rc = "", -99
            print(f"  [warn] fm call {i} raised: {e}", file=sys.stderr)
        got = parse_letter(text)
        is_correct = got == expected
        if is_correct:
            correct += 1
        by_subject[q["subject"]]["n"] += 1
        if is_correct:
            by_subject[q["subject"]]["correct"] += 1
        samples.append({"i": i, "subject": q["subject"], "expected": expected,
                        "got": got, "correct": bool(is_correct)})
        if verbose:
            print(f"  [{i}] {q['subject'][:24]:24} expected={expected} "
                  f"got={got} correct={is_correct}"
                  + ("" if rc == 0 else f" (rc={rc})"))
        if (i + 1) % progress_every == 0:
            acc = correct / (i + 1)
            el = time.monotonic() - t0
            print(f"  MMLU {i+1}/{len(sample)}  acc={acc:.3f}  elapsed={el:.0f}s", flush=True)
    runtime_s = time.monotonic() - t0
    n = len(sample)
    accuracy = correct / n if n else 0.0
    by_subject_out = OrderedDict()
    for s in subjects:
        d = by_subject[s]
        if d["n"] > 0:
            by_subject_out[s] = {
                "n": d["n"],
                "correct": d["correct"],
                "accuracy": round(d["correct"] / d["n"], 4),
            }
    out = {
        "suite": "MMLU",
        "source": "cais/mmlu (test)",
        "license": "MIT",
        "n": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_subject": by_subject_out,
        "runtime_s": round(runtime_s, 1),
    }
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "spot":
        # Stratified sample but only run the first 5 in file order
        rows = load_all()
        sample, _ = stratified_sample(rows)
        correct = 0
        for i, q in enumerate(sample[:5]):
            expected = q["answer"].strip().upper()
            res = F.respond(build_prompt(q), greedy=True)
            got = parse_letter(res["text"])
            ok = got == expected
            correct += ok
            print(f"  [{i}] {q['subject'][:24]:24} expected={expected} "
                  f"got={got} correct={ok}")
        print(json.dumps({"spot_n": 5, "correct": correct}, indent=2))
        print(f"Total stratified sample size: {len(sample)}")
        return
    out = run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    F.write_json(os.path.join(RESULTS_DIR, "mmlu.json"), out)
    print(f"MMLU DONE: {out['correct']}/{out['n']} = {out['accuracy']:.4f} "
          f"({len(out['by_subject'])} subjects) in {out['runtime_s']}s")


if __name__ == "__main__":
    main()

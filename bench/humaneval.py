"""HumanEval pass@1 (n=1) benchmark for Apple Foundation Models (`fm` CLI).

All 164 tasks, greedy. The model is given the function signature + docstring
(`task["prompt"]`) and asked to output only a single ```python code block. We
extract that block, de-duplicate a re-emitted `def <entry_point>` signature,
then assemble `prompt + completion + "\n" + test + "\ncheck(entry_point)\n"` and
execute it in a SANDBOXED subprocess (10s timeout, fresh temp cwd, no network env
vars, PYTHONPATH cleared). pass = exit code 0.
"""
from __future__ import annotations
import sys, os, re, json, time, subprocess, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm_common as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "humaneval.jsonl")
RESULTS_DIR = os.path.join(HERE, "..", "results")
EXEC_TIMEOUT = 10  # seconds per generated program


def load():
    rows = []
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(task) -> str:
    sig = task["prompt"]
    return (
        "Complete the following Python function. Output ONLY the function body / "
        "completion as a single ```python code block. Do not include tests or "
        "explanations.\n\n"
        "```python\n"
        + sig
        + "\n```"
    )


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, entry_point: str, sig: str) -> str:
    """Pull the first fenced python block; if none, fall back to raw text.

    Returns a code string whose lines carry their *natural* indentation, ready to
    append after sig (which ends with the closing triple-quote of the docstring at
    the function-body indent). We strip only leading/trailing BLANK lines — never
    the per-line leading spaces, because the function body must stay at 4-space
    indent.
    """
    code = None
    if text:
        m = _FENCE.search(text)
        if m:
            code = m.group(1)
    if code is None:
        code = text or ""

    lines = code.splitlines()
    # drop leading/trailing blank lines but keep internal indentation intact
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    code = "\n".join(lines)
    if not code:
        return code

    # If the model re-emitted the WHOLE function (starts with `def <entry_point>`),
    # treat its output as the full implementation: replace sig entirely.
    if re.match(rf"\s*(from\s+\S+\s+import|import\s+\S+|def\s+{re.escape(entry_point)}\s*\()",
                code):
        return code  # caller will use this as a standalone-ish block

    # De-duplicate a re-emitted signature line at the top of the body.
    sig_def = (re.search(rf"(^[ \t]*def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:\s*->.*$)",
                         sig, re.MULTILINE)
               or re.search(rf"(^[ \t]*def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:\s*$)",
                            sig, re.MULTILINE))
    if sig_def is not None:
        sig_line = sig_def.group(1).rstrip()
        blines = code.splitlines()
        if blines and blines[0].rstrip() == sig_line:
            code = "\n".join(blines[1:])
    return code


def sandbox_env():
    """Minimal env: keep PATH/HOME/TMP only, clear PYTHONPATH and net vars."""
    keep = {}
    for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "SHELL"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep["PYTHONPATH"] = ""   # cleared
    keep["PYTHONNOUSERSITE"] = "1"
    return keep


def run_program(full_code: str):
    """Execute assembled program in an isolated subprocess. Returns (passed, err)."""
    workdir = tempfile.mkdtemp(prefix="humaneval_")
    path = os.path.join(workdir, "sol.py")
    try:
        with open(path, "w") as f:
            f.write(full_code)
        try:
            p = subprocess.run(
                [sys.executable, path],
                cwd=workdir,
                env=sandbox_env(),
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout({EXEC_TIMEOUT}s)"
        if p.returncode == 0:
            return True, None
        err = (p.stderr or "").strip()
        # keep error concise but informative
        err_lines = err.splitlines()
        # find the traceback's final exception line
        last = ""
        for ln in err_lines:
            if ln.strip() and not ln.startswith(" "):
                last = ln.strip()
        return False, (last[-200:] if last else (err[-200:] or f"exit {p.returncode}"))
    except Exception as e:
        return False, f"harness error: {e}"
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass


def run(verbose=False, progress_every=25):
    rows = load()
    samples = []
    passed_n = 0
    t0 = time.monotonic()
    for i, task in enumerate(rows):
        entry = task["entry_point"]
        sig = task["prompt"]
        prompt = build_prompt(task)
        err = None
        try:
            res = F.respond(prompt, greedy=True)
            text = res["text"]
            if res["returncode"] != 0:
                err = f"fm rc={res['returncode']} {res.get('stderr','')[:80]}"
                text = text or ""
        except Exception as e:
            text, err = "", f"fm raised: {e}"
        completion = extract_code(text, entry, sig)
        if not completion:
            ok, e = False, (err or "empty completion")
        else:
            starts_with_def = bool(re.match(
                rf"\s*(def\s+{re.escape(entry)}\s*\()",
                completion))
            if starts_with_def:
                # Model returned the whole function (and maybe imports above it).
                # Use it as-is; don't prepend sig.
                full = completion + "\n" + task["test"] + f"\ncheck({entry})\n"
            else:
                full = sig + completion + "\n" + task["test"] + f"\ncheck({entry})\n"
            ok, e = run_program(full)
            if not ok and err is None:
                err = e
        if ok:
            passed_n += 1
        samples.append({
            "task_id": task["task_id"],
            "entry_point": entry,
            "passed": bool(ok),
            "error": (err if not ok else None),
        })
        if verbose:
            print(f"  [{task['task_id']}] {entry} passed={ok} "
                  + ("" if ok else f"err={err}"))
        if (i + 1) % progress_every == 0:
            el = time.monotonic() - t0
            print(f"  HumanEval {i+1}/{len(rows)}  pass@1={passed_n/(i+1):.3f}  "
                  f"elapsed={el:.0f}s", flush=True)
    runtime_s = time.monotonic() - t0
    n = len(rows)
    pass_at_1 = passed_n / n if n else 0.0
    out = {
        "suite": "HumanEval",
        "source": "openai/human-eval",
        "license": "MIT",
        "n": n,
        "passed": passed_n,
        "pass_at_1": round(pass_at_1, 4),
        "runtime_s": round(runtime_s, 1),
        "samples": samples,
    }
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "spot":
        rows = load()[:5]
        passed_n = 0
        for task in rows:
            entry = task["entry_point"]; sig = task["prompt"]
            res = F.respond(build_prompt(task), greedy=True)
            comp = extract_code(res["text"], entry, sig)
            if not comp:
                ok, e = False, "empty completion"
            else:
                starts_with_def = bool(re.match(rf"\s*(def\s+{re.escape(entry)}\s*\()", comp))
                if starts_with_def:
                    full = comp + "\n" + task["test"] + f"\ncheck({entry})\n"
                else:
                    full = sig + comp + "\n" + task["test"] + f"\ncheck({entry})\n"
                ok, e = run_program(full)
            passed_n += ok
            print(f"  [{task['task_id']}] {entry} passed={ok} "
                  + ("" if ok else f"err={e}"))
        print(json.dumps({"spot_n": len(rows), "passed": passed_n}, indent=2))
        return
    out = run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    F.write_json(os.path.join(RESULTS_DIR, "humaneval.json"), out)
    print(f"HumanEval DONE: {out['passed']}/{out['n']} pass@1={out['pass_at_1']:.4f} "
          f"in {out['runtime_s']}s")


if __name__ == "__main__":
    main()

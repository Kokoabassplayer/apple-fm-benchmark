"""Shared helpers for the Apple Foundation Models (`fm`) CLI benchmark.

All measurement modules import from here so the call interface, ANSI stripping,
token counting, and timing are identical everywhere.

Key honesty notes baked into the helpers:
  * `fm` emits ANSI SGR color codes in its stdout — we strip them everywhere,
    so token counts and answer matching are not corrupted by escape bytes.
  * Timing uses time.monotonic() (immune to wall-clock jumps).
  * respond_stream() returns BOTH the time-to-first-token (prefill-dominated)
    and the total elapsed, so callers can separate prefill from decode.
"""
from __future__ import annotations
import os, re, subprocess, time, json

MODEL = os.environ.get("FM_MODEL", "system")          # on-device 3B
FM_BIN = os.environ.get("FM_BIN", "/usr/bin/fm")
DEFAULT_TIMEOUT = 200

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(s: str) -> str:
    """Remove ALL ANSI escape sequences (SGR colors, cursor moves, etc.)."""
    return _ANSI.sub("", s)


def tok_count(s: str) -> int:
    """Token count via `fm token-count`. Returns 0 if it can't be parsed."""
    p = subprocess.run([FM_BIN, "token-count", s],
                       capture_output=True, text=True, timeout=60)
    m = re.search(r"(\d+)", strip_ansi(p.stdout))
    return int(m.group(1)) if m else 0


def respond(prompt: str, *, greedy: bool = True, stream: bool = False,
            instructions: str | None = None, timeout: int = DEFAULT_TIMEOUT,
            model: str = MODEL):
    """Blocking call. Returns dict(text, wall_s, returncode, stderr).

    Used where we only need the final text + total wall time.
    """
    args = [FM_BIN, "respond", "--model", model]
    if greedy:
        args.append("--greedy")
    args.append("--stream" if stream else "--no-stream")
    if instructions is not None:
        args += ["--instructions", instructions]
    t0 = time.monotonic()
    try:
        p = subprocess.run(args, input=prompt, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"text": "", "wall_s": float(timeout), "returncode": -1,
                "stderr": "timeout", "ttft": None}
    dt = time.monotonic() - t0
    return {"text": strip_ansi(p.stdout).strip(),
            "wall_s": dt, "returncode": p.returncode,
            "stderr": strip_ansi(p.stderr).strip()[:200], "ttft": None}


def respond_stream(prompt: str, *, greedy: bool = True,
                   instructions: str | None = None,
                   timeout: int = DEFAULT_TIMEOUT, model: str = MODEL):
    """Streamed call. Returns dict(text, ttft_s, wall_s, returncode, stderr).

    ttft_s = time from process start to the first chunk of *content* bytes
    arriving on stdout (prefill + spawn + 1-token decode). The slope of ttft
    across input sizes is the prefill rate; decode rate comes from
    (wall - ttft) over the output token count.
    """
    args = [FM_BIN, "respond", "--model", model]
    if greedy:
        args.append("--greedy")
    args.append("--stream")
    if instructions is not None:
        args += ["--instructions", instructions]
    t0 = time.monotonic()
    ttft = None
    chunks = []
    try:
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, bufsize=1)
        # write the prompt then close stdin so the model begins inference
        try:
            p.stdin.write(prompt)
            p.stdin.close()
        except BrokenPipeError:
            pass
        assert p.stdout is not None
        for line in p.stdout:
            if ttft is None and strip_ansi(line).strip():
                ttft = time.monotonic() - t0
            chunks.append(line)
        p.wait(timeout=timeout)
        rc = p.returncode
        err = strip_ansi((p.stderr.read() if p.stderr else "") or "").strip()[:200]
    except subprocess.TimeoutExpired:
        p.kill()
        return {"text": "".join(chunks), "ttft": ttft, "wall_s": float(timeout),
                "returncode": -1, "stderr": "timeout"}
    dt = time.monotonic() - t0
    return {"text": strip_ansi("".join(chunks)).strip(), "ttft": ttft,
            "wall_s": dt, "returncode": rc, "stderr": err}


def stats(xs):
    """median, mean, p95, std, min, max, n — None-safe for tiny lists."""
    import statistics
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0, "median": None, "mean": None, "p95": None,
                "std": None, "min": None, "max": None}
    n = len(xs)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "n": n,
        "median": round(xs[n // 2], 3),
        "mean": round(statistics.mean(xs), 3),
        "p95": round(xs[p95_idx], 3),
        "std": round(statistics.pstdev(xs), 3) if n > 1 else 0.0,
        "min": round(xs[0], 3),
        "max": round(xs[-1], 3),
    }


def write_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  wrote {path}")


# Benign filler for building prompts of a target token length (the Brindleford
# essay from context_window.py — guardrail-safe, no "ignore above" framing).
ESSAY = (
    "Brindleford is a quiet market town set among low green hills in the north. "
    "Its narrow streets follow the curve of an old cattle-droving route, and the "
    "stone cottages along the high street date back several centuries. Each "
    "Tuesday a small market opens in the square, selling bread, cheese, wool, and "
    "seasonal vegetables from the surrounding farms. The townsfolk are known for "
    "keeping neat gardens and for a stubborn preference for tea over coffee. "
    "A river, shallow and clear, winds past the eastern edge of the town, and a "
    "footbridge of weathered oak connects the lower meadow to the mill road. "
    "Children learn to read in a single schoolhouse, and the bell there marks "
    "the hours of the day. Travellers often remark on the calm of the place. "
)


def filler_of(target_tokens: int) -> str:
    """Build ~target_tokens of benign filler text."""
    base = tok_count(ESSAY) or 50
    reps = max(1, target_tokens // base + 2)
    blob = ESSAY * reps
    cnt = tok_count(blob)
    if cnt > 0:
        blob = blob[: int(len(blob) * target_tokens / cnt)]
    return blob

# Apple Foundation Models — On-Device Benchmark

A reproducible, spec-labeled benchmark of the **on-device** language model Apple
ships in **macOS 27** (Apple Foundation Models, called via the `fm` CLI). No API
key, no cloud, no per-token bill — every number is measured on one Mac and you
can re-measure it on yours.

🔗 **Live dashboard:** <https://kokoabassplayer.github.io/apple-fm-benchmark>

> Measured on **MacBook Pro · Apple M3 Pro · 11 cores (5P+6E) · 18 GB · macOS 27.0 (build 26A5353q)**.

## Headline findings (this machine)

| | Value | How |
|---|---|---|
| **Usable context** | ~4,000 tokens; 100% mid-context recall to 3k | binary-search ceiling + multi-needle recall |
| **Prefill** (reads prompt) | ~967 tok/s | slope of TTFT vs input length |
| **Decode** (writes answer) | ~71 tok/s | slope of wall-time vs output length |
| **Warm first-token** | ~0.42 s | median, short output |
| **Cold first-token** | ~2.44 s | first call (model load) |
| **Network privacy** | 0 model-related processes observed | differential en0 bytes + nettop, during a 23s generation |
| **GSM8K / HumanEval / MMLU** | see dashboard | standard open suites |

A single "tokens/sec" number is misleading, so we report **two** speeds (prefill
and decode), each derived by **linear regression** — not a stopwatch.

## What's in this repo

```
index.html              # the dashboard (self-contained; data embedded + fetches results/all.json)
README.md  README.th.md # this file, in English and Thai
LICENSE                 # MIT (the harness + dashboard)
bench/
  spec.sh      latency.py   context.py   privacy.py    # measurement modules
  gsm8k.py     humaneval.py  mmlu.py                    # standard accuracy suites
  run_all.py                                           # orchestrator -> results/all.json
  fm_common.py                                         # shared fm CLI helpers
  DATA_LICENSES.md                                     # dataset licenses + HumanEval sandbox note
results/                # raw JSON: spec, latency, context, privacy, gsm8k, humaneval, mmlu, all.json
social/                 # Thai social thread + carousel slides
```

## Reproduce it on your Mac

Requirements: macOS 27 with the `fm` CLI present (`/usr/bin/fm`), Python 3.10+,
and `git`. (Apple Foundation Models are macOS 27+ only.)

```bash
git clone https://github.com/kokoabassplayer/apple-fm-benchmark
cd apple-fm-benchmark
bash bench/spec.sh                 # capture your machine spec
python3 bench/run_all.py           # latency + context + privacy + 3 accuracy suites (~1 hour)
```

Then open `index.html` (or commit your `results/all.json` and the dashboard
updates automatically — it fetches that file on load).

> **HumanEval runs model-generated code.** It executes in an isolated subprocess
> (10s timeout, fresh temp dir, no network, no elevated privileges). See
> `bench/DATA_LICENSES.md`.

## How this is measured (so you can trust it)

- **Spec-labeled.** Every number is tied to the machine in the spec strip. A
  different chip will differ — that's the point.
- **Regression, not stopwatch.** Prefill = slope of time-to-first-token vs input
  length; decode = slope of wall-time vs output length. Both fit across multiple
  sizes so a single noisy run can't skew them.
- **Context = can-FIT vs can-RETRIEVE.** A binary-searched ceiling (fits) plus a
  multi-needle recall curve (does it actually find facts in the middle?).
- **Privacy is evidence, not proof.** Differential en0 byte-delta + full `nettop`
  process list during inference vs idle. We observed zero model-related traffic;
  *proving* fully-offline needs the Wi-Fi-off test (a manual step).
- **Measured vs reference split.** Amber = measured on this Mac. Grey =
  vendor-stated marketing figures, included for scale only.
- **"3B" is not asserted.** Apple does not publish the parameter count.

## Add your machine

This is most useful as a shared dataset. Run the harness on your Mac, then open a
PR adding your `results/spec.json` + headline numbers (or your full
`results/*.json`). Over time the README/dashboard can grow a cross-chip table
(M1 / M2 / M3 / M4 …).

## License & attribution

- Harness + dashboard: **MIT** (this repo).
- Datasets: GSM8K (MIT), HumanEval (MIT), MMLU (MIT; individual questions carry
  mixed origins incl. non-commercial).
- Not affiliated with Apple. Independent, reproducible measurement.

Thai version: **[README.th.md](README.th.md)** · แบบไทยดูที่ `README.th.md`

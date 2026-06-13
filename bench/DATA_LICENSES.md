# Dataset Licenses

All datasets are cached under `bench/data/` and used only for offline accuracy
benchmarking of Apple's on-device Foundation Models (`fm` CLI).

| Dataset   | Source                                              | License                                                                                       | Notes |
|-----------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------|-------|
| GSM8K     | `openai/grade-school-math` (test split)             | MIT                                                                                           | 1319 grade-school math word problems. Ground truth = integer after `#### `. |
| HumanEval | `openai/human-eval`                                 | MIT                                                                                           | 164 hand-written Python function-completion tasks. pass@1 (n=1). |
| MMLU      | `cais/mmlu` (test split)                            | MIT (code/data release) — individual questions carry mixed origins incl. CC-BY-NC-SA; treat downstream content as **non-commercial** | 14042 multiple-choice questions across 57 subjects. |

## How each was obtained
- **GSM8K**: `curl` from `raw.githubusercontent.com/openai/grade-school-math/.../test.jsonl`.
- **HumanEval**: `curl` of `HumanEval.jsonl.gz` from `openai/human-eval`, then `gunzip`.
- **MMLU**: The spec's CSV URL (`.../data/test-00000-of-00001.csv`) currently 404s; the
  CAIS repo stores the test split as **parquet** at `all/test-00000-of-00001.parquet`.
  We downloaded that parquet and converted it locally to a CSV with columns
  `question,A,B,C,D,answer,subject` (where `answer` is the letter A–D). The underlying
  data is identical to the MMLU test split.

## Sandboxing note (HumanEval)
HumanEval scoring executes model-generated Python in an isolated subprocess:
10s timeout, fresh temp working directory, an environment with network-related
variables removed and `PYTHONPATH` cleared, never with elevated privileges.
Failures (timeouts, exceptions, non-zero exit) are scored as fail — this is the
fair pass@1 (n=1) convention.

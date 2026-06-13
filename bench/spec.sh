#!/usr/bin/env bash
# Capture the machine + toolchain spec for the benchmark. Every published number
# is labeled with this spec, because on-device performance is meaningless
# without it (an M1 and an M3 Max are not the same "Mac").
#
# Output: results/spec.json
set -u
cd "$(dirname "$0")/.."
mkdir -p results

# --- hardware (system_profiler is the authoritative source on macOS) ---
hw=$(system_profiler SPHardwareDataType)
get() { echo "$hw" | grep -E "^ *$1:" | sed -E "s/^ *$1: *//" | tr -d '\r'; }
model_name=$(get "Model Name")
model_id=$(get "Model Identifier")
chip=$(get "Chip")
cores_line=$(get "Total Number of Cores")
ram=$(get "Memory")

# parse "11 (5 Performance and 6 Efficiency)"
total_cores=$(echo "$cores_line" | grep -oE '^[0-9]+' || echo "")
perf_cores=$(echo "$cores_line" | grep -oE '[0-9]+ Performance' | grep -oE '^[0-9]+' || echo "")
eff_cores=$(echo "$cores_line" | grep -oE '[0-9]+ Efficiency' | grep -oE '^[0-9]+' || echo "")

# --- macOS ---
macos_product=$(sw_vers -productName)
macos_version=$(sw_vers -productVersion)
macos_build=$(sw_vers -buildVersion)

# --- fm CLI (no --version flag exists; record path + build mtime + size) ---
fm_bin=$(command -v fm || echo /usr/bin/fm)
fm_mtime=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$fm_bin" 2>/dev/null)
fm_size=$(stat -f "%z" "$fm_bin" 2>/dev/null)
fm_avail=$(fm available 2>&1 | tr -d '\r')

# --- python ---
py=$(python3 --version 2>&1)

# --- measured-at (UTC, stable for repro) ---
when=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

python3 - "$model_name" "$model_id" "$chip" "$total_cores" "$perf_cores" \
           "$eff_cores" "$ram" "$macos_product" "$macos_version" "$macos_build" \
           "$fm_bin" "$fm_mtime" "$fm_size" "$fm_avail" "$py" "$when" <<'PY'
import json, sys, re
(_model_name,_model_id,_chip,_total_cores,_perf_cores,_eff_cores,_ram,
 _mac_product,_mac_version,_mac_build,_fm_bin,_fm_mtime,_fm_size,
 _fm_avail,_py,_when) = sys.argv[1:]
_fm_avail = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", _fm_avail)
spec = {
  "machine": {
    "model_name": _model_name,
    "model_identifier": _model_id,
    "chip": _chip,
    "cores_total": _total_cores,
    "cores_performance": _perf_cores,
    "cores_efficiency": _eff_cores,
    "memory": _ram,
  },
  "os": {"product": _mac_product, "version": _mac_version, "build": _mac_build},
  "fm_cli": {
    "path": _fm_bin,
    "build_mtime_utc_guess": _fm_mtime,
    "size_bytes": int(_fm_size) if _fm_size.isdigit() else None,
    "note": "fm has no --version flag; binary path + mtime + macOS build identify it",
    "availability": _fm_avail.strip(),
  },
  "runtime": {"python": _py},
  "measured_at_utc": _when,
}
with open("results/spec.json", "w") as f:
    json.dump(spec, f, indent=2, ensure_ascii=False)
print(json.dumps(spec, indent=2, ensure_ascii=False))
PY

#!/usr/bin/env python3
"""Network-privacy evidence for the on-device `fm` model.

We CANNOT prove "zero bytes leave the machine" with a passive measurement —
background services (iCloud, mDNS, push, NTP) move bytes constantly. Instead we
collect DIFFERENTIAL evidence:

  * en0 outbound byte-delta during an inference window vs an equal idle window.
    excess = call_delta - idle_delta  (bytes attributable to the activity window)
  * Full per-process enumeration (nettop, external traffic only) during BOTH
    windows, so a reader can see exactly which processes moved external bytes
    and whether anything model/fm-related appears.

The honest conclusion is framed as "no model-related external traffic observed
in this window; differential excess = X bytes" — NOT "provably offline".
Proving offline requires physically disabling the network (manual wifi test,
requested separately).

Outputs results/privacy.json. Stdlib only.
"""
import subprocess, time, re, os
import fm_common as F

GEN_PROMPT = ("Count from 1 to 400. Output the numbers separated by single "
              "spaces. No words, no punctuation, no commentary.")
NETTOP_SAMPLES = 45           # ~1s each; covers a ~24s generation + margin


def en0_obytes():
    """en0 outbound bytes from netstat -ib (Obytes on the <Link# row)."""
    p = subprocess.run(["netstat", "-ib"], capture_output=True, text=True, timeout=15)
    for line in p.stdout.splitlines():
        c = line.split()
        if len(c) >= 10 and c[0] == "en0" and "<Link" in line:
            try:
                return int(c[9])
            except ValueError:
                pass
    return None


def start_nettop():
    """Start nettop (external, per-process). It exits on its own after -L samples,
    which guarantees its stdout buffer flushes — terminating mid-run loses data."""
    args = ["nettop", "-P", "-x", "-J", "bytes_in,bytes_out",
            "-t", "external", "-L", str(NETTOP_SAMPLES)]
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True)


def parse_processes(nettop_text):
    """Parse nettop CSV (time,proc.pid,bytes_in,bytes_out,).

    Counters are cumulative since process start, so per-window DELTA =
    last_sample - first_sample for each process. Returns {name: delta_in/out}.
    """
    first, last = {}, {}
    for line in nettop_text.splitlines():
        f = line.split(",")
        if len(f) < 4:
            continue
        name_pid = f[1].strip()
        if not name_pid or name_pid in ("", "time"):
            continue
        try:
            bin_ = int(f[2]); bout = int(f[3])
        except ValueError:
            continue
        name = re.split(r"\.\d+$", name_pid)[0].strip()
        if name not in first:
            first[name] = (bin_, bout)
        last[name] = (bin_, bout)
    agg = {}
    for name, (bi0, bo0) in first.items():
        bi1, bo1 = last[name]
        din, dout = bi1 - bi0, bo1 - bo0
        if din > 0 or dout > 0:
            agg[name] = {"bytes_in_delta": din, "bytes_out_delta": dout}
    return agg


def run_window(label, with_gen):
    print(f"[privacy] {label} window (gen={with_gen})")
    b0 = en0_obytes()
    p = start_nettop()                    # exits on its own after NETTOP_SAMPLES
    gen_wall = None
    if with_gen:
        t0 = time.monotonic()
        F.respond(GEN_PROMPT)
        gen_wall = time.monotonic() - t0
    else:
        time.sleep(24)
    nt, _ = p.communicate(timeout=60)     # natural exit => flushed buffer
    b1 = en0_obytes()
    delta = (b1 - b0) if (b0 is not None and b1 is not None) else None
    procs = parse_processes(nt)
    flagged = [n for n in procs if re.search(r"fm|foundation|model|apple intelligence",
                                             n, re.I)]
    return {"bytes_delta_en0_out": delta, "gen_wall_s": round(gen_wall, 2)
            if gen_wall else None, "processes": procs, "flagged_model_related": flagged,
            "process_count": len(procs)}


def main():
    call = run_window("inference", True)
    time.sleep(2)
    idle = run_window("idle", False)
    excess = None
    if call["bytes_delta_en0_out"] is not None and idle["bytes_delta_en0_out"] is not None:
        excess = call["bytes_delta_en0_out"] - idle["bytes_delta_en0_out"]
    out = {
        "inference_window": call,
        "idle_window": idle,
        "differential_excess_bytes_out": excess,
        "honest_summary": {
            "model_related_processes_seen": call["flagged_model_related"],
            "excess_vs_idle": excess,
            "interpretation": (
                "Differential en0 outbound bytes = inference-window minus "
                "idle-window. A small/zero excess with NO model-related process "
                "in nettop is evidence (not proof) of on-device privacy. Proving "
                "offline requires disabling the network (manual wifi test)."),
        },
        "methodology": (
            f"netstat -ib en0 Obytes delta + nettop -P -x -J bytes_in,bytes_out "
            f"-t external -L {NETTOP_SAMPLES} during a ~24s greedy generation vs "
            f"equal idle. en0 bytes are total-interface (background services "
            f"contribute); only the differential excess is attributable to the "
            f"activity window."),
    }
    F.write_json("results/privacy.json", out)
    print(json_summary(out))


def json_summary(o):
    c, i = o["inference_window"], o["idle_window"]
    return (f"\n inference out-delta: {c['bytes_delta_en0_out']} bytes "
            f"({len(c['processes'])} procs)\n"
            f" idle      out-delta: {i['bytes_delta_en0_out']} bytes "
            f"({len(i['processes'])} procs)\n"
            f" excess: {o['differential_excess_bytes_out']} bytes; "
            f"model-related procs: {c['flagged_model_related']}")


if __name__ == "__main__":
    main()

"""Minimal TUI for live experiment monitoring on Windows cmd.

Reads checkpoints + log tails, no external dependencies.
Run: python monitor_tui.py
Quit: Ctrl+C
"""

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "raw_results"
LOGS = ROOT / "logs"

# Cells we care about
PHASE_DEFS = {
    "Tier 4a oracle+random":     ("exp_oracle_random",        600),
    "Tier 4b DRQN det":          ("exp_memory_agents",         20),
    "Tier 4b DRQN nondet":       ("exp_memory_agents_nondet",  20),
    "Phase 3A V2 tabular":       ("exp_v2_tabular",           240),
    "Phase 3B capacity":         ("exp_capacity_study",       160),
    "Tier 2 K4 ablation":        ("exp_reward_ablation_fast", 200),
}

REFRESH_S = 4

# ANSI color codes (Windows 10+ cmd supports them after VT mode enable)
RESET = "\x1b[0m"
BOLD  = "\x1b[1m"
GREEN = "\x1b[32m"
YEL   = "\x1b[33m"
CYAN  = "\x1b[36m"
RED   = "\x1b[31m"
DIM   = "\x1b[2m"
MAG   = "\x1b[35m"


def enable_vt() -> None:
    """Enable VT/ANSI mode on Windows cmd by writing a known escape sequence."""
    if sys.platform == "win32":
        try:
            subprocess.run(["cmd", "/c", "rem"], check=False, timeout=2)
        except Exception:
            pass


def clear() -> None:
    # ANSI clear screen + cursor home
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def count_results(subdir: str) -> tuple[int, int]:
    """Returns (json_files_count, checkpoint_entries)."""
    d = RESULTS / subdir
    if not d.exists():
        return (0, 0)
    files = len(list(d.glob("*.json"))) - (1 if (d / "checkpoint.json").exists() else 0)
    cp = d / "checkpoint.json"
    n_cp = 0
    if cp.exists():
        try:
            n_cp = len(json.loads(cp.read_text()))
        except Exception:
            n_cp = 0
    return (max(files, 0), n_cp)


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip()
        util, used, total = [int(x.strip()) for x in out.split(",")]
        return f"GPU {util:>2}%   VRAM {used:>5}/{total:>5} MiB"
    except Exception:
        return "GPU n/a"


def python_procs() -> list[tuple[int, str]]:
    """Returns list of (pid, script_name) for running python processes."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python3.11.exe' or Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            text=True, timeout=4,
        ).strip()
        if not out:
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        out_list = []
        for p in data:
            pid = p.get("ProcessId", 0)
            cmd = p.get("CommandLine") or ""
            script = ""
            for tok in cmd.split():
                if tok.endswith(".py"):
                    script = Path(tok).name
                    break
            if script:
                out_list.append((pid, script))
        return out_list
    except Exception:
        return []


def tail_log(path: Path, n: int = 3) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
        return [l.rstrip() for l in lines[-n:] if l.strip()]
    except Exception:
        return []


def newest_log_in(pattern: str) -> Path | None:
    matches = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def render_bar(done: int, total, width: int = 30) -> str:
    if total is None or total == 0:
        return f"[{DIM}{'-' * width}{RESET}] {done:>3}/?"
    pct = min(done / total, 1.0)
    fill = int(pct * width)
    bar = "#" * fill + "-" * (width - fill)
    color = GREEN if pct >= 1.0 else (YEL if pct >= 0.5 else CYAN)
    return f"[{color}{bar}{RESET}] {done:>3}/{total} {color}{int(pct * 100):>3}%{RESET}"


def capacity_breakdown() -> str:
    d = RESULTS / "exp_capacity_study"
    cp = d / "checkpoint.json"
    if not cp.exists():
        return ""
    try:
        items = json.loads(cp.read_text())
    except Exception:
        return ""
    by = defaultdict(lambda: defaultdict(int))
    for k in items:
        # MLP_DQN_h32_9_42
        parts = k.split("_")
        if len(parts) >= 5 and parts[2].startswith("h"):
            cap = parts[2]
            size = parts[3]
            by[cap][size] += 1
    if not by:
        return ""
    rows = []
    for cap in ("h32", "h64", "h128", "h256"):
        c = by.get(cap, {})
        s9 = c.get("9", 0)
        s13 = c.get("13", 0)
        rows.append(f"{cap:>5}: 9x9={s9:>2}/20  13x13={s13:>2}/20")
    return "\n  ".join(rows)


def main() -> None:
    enable_vt()
    start = time.time()
    last_progress: dict[str, int] = {}
    last_progress_time: dict[str, float] = {}

    try:
        while True:
            clear()
            now = time.time()
            elapsed = int(now - start)

            print(f"{BOLD}{CYAN}=== Engram experiment monitor ==={RESET}  "
                  f"{DIM}refresh {REFRESH_S}s  uptime {elapsed//60}m{elapsed%60:02d}s  Ctrl+C to quit{RESET}")
            print()

            # Phase progress block
            print(f"{BOLD}Phase progress{RESET}")
            for label, (subdir, target) in PHASE_DEFS.items():
                files, cp = count_results(subdir)
                done = max(files, cp)
                bar = render_bar(done, target)
                # Speed indicator
                key = subdir
                speed = ""
                if key in last_progress and last_progress[key] != done:
                    delta = done - last_progress[key]
                    dt = now - last_progress_time[key]
                    if dt > 0 and delta > 0:
                        rate = delta / dt
                        speed = f"  {GREEN}+{delta} ({rate*60:.1f}/min){RESET}"
                if key not in last_progress or last_progress[key] != done:
                    last_progress[key] = done
                    last_progress_time[key] = now
                print(f"  {label:<24} {bar}{speed}")
            print()

            # Capacity breakdown (Phase 3B detail)
            cap = capacity_breakdown()
            if cap:
                print(f"{BOLD}Phase 3B capacity detail{RESET}")
                print(f"  {cap}")
                print()

            # Running processes
            procs = python_procs()
            print(f"{BOLD}Running python processes{RESET}")
            if procs:
                for pid, script in procs:
                    print(f"  {GREEN}PID {pid:>6}{RESET}  {script}")
            else:
                print(f"  {DIM}(none){RESET}")
            print()

            # GPU
            print(f"{BOLD}{gpu_status()}{RESET}")
            print()

            # Latest log lines
            for label, pat in [
                ("Phase 3B latest log", "phase3b_resume*.log"),
                ("DRQN latest log",     "tier4b_drqn_resume*.log"),
            ]:
                p = newest_log_in(pat)
                if p:
                    lines = tail_log(p, 2)
                    if lines:
                        print(f"{BOLD}{label}{RESET}  {DIM}{p.name}{RESET}")
                        for l in lines:
                            print(f"  {l[:120]}")
                        print()

            # Sleep till next refresh
            time.sleep(REFRESH_S)
    except KeyboardInterrupt:
        print(f"\n{YEL}monitor stopped.{RESET}")


if __name__ == "__main__":
    main()

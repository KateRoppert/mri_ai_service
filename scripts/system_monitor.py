#!/usr/bin/env python3
"""
System RAM/swap monitor — host-side OOM watchdog for pipeline load testing.

Runs on the HOST (not inside Docker) so it observes total system memory,
which is what the kernel OOM killer acts on. Complements gpu_monitor.py
(GPU) and performance_monitor.py (per-stage benchmarks).

Dependency-free: reads /proc/meminfo directly (Linux only). Optionally scans
/proc/<pid>/status for the top memory consumers so you can see WHICH process
(which pipeline worker) is driving usage.

USAGE:
    # Run in a separate terminal alongside the pipeline:
    python scripts/system_monitor.py --interval 2 --top 5 \
        --output demo_workspace/input/<run>/logs/system_monitor.csv

    # Stop with Ctrl+C — prints a peak-usage summary.

WHY THIS EXISTS:
    Running the pipeline on all 3 SibBMS patients (11 sessions, 33 volumes)
    with Stage 04 quality at 12 workers exhausted 30 GB RAM + 8 GB swap in
    ~33 s. The kernel OOM killer took down the desktop (VSCode, Firefox).
    This tool measures real peak RAM/swap per stage so worker counts can be
    tuned from data instead of guesses.
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_meminfo() -> Dict[str, int]:
    """Parse /proc/meminfo into a dict of {key: kB_value}."""
    info: Dict[str, int] = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            # Lines look like: "MemTotal:       32797156 kB"
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value_str = parts[1].strip().split()[0]  # drop the "kB" suffix
            try:
                info[key] = int(value_str)
            except ValueError:
                continue
    return info


def compute_usage(info: Dict[str, int]) -> Dict[str, float]:
    """Derive RAM/swap usage (GB and percent) from a meminfo dict."""
    mem_total_kb = info.get("MemTotal", 0)
    # MemAvailable is the kernel's best estimate of allocatable memory
    # without swapping — a better "free" signal than MemFree alone.
    mem_avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
    mem_used_kb = mem_total_kb - mem_avail_kb

    swap_total_kb = info.get("SwapTotal", 0)
    swap_free_kb = info.get("SwapFree", 0)
    swap_used_kb = swap_total_kb - swap_free_kb

    def gb(kb: int) -> float:
        return kb / (1024.0 * 1024.0)

    return {
        "ram_total_gb": gb(mem_total_kb),
        "ram_used_gb": gb(mem_used_kb),
        "ram_pct": (mem_used_kb / mem_total_kb * 100.0) if mem_total_kb else 0.0,
        "swap_total_gb": gb(swap_total_kb),
        "swap_used_gb": gb(swap_used_kb),
        "swap_pct": (swap_used_kb / swap_total_kb * 100.0) if swap_total_kb else 0.0,
    }


def top_processes(n: int) -> List[Tuple[int, str, float]]:
    """Return the top-n processes by resident memory (RSS) as (pid, name, rss_gb).

    Scans /proc/<pid>/status. Best-effort: processes that exit mid-scan or
    that we can't read are skipped silently.
    """
    procs: List[Tuple[int, str, float]] = []
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit():
            continue
        status_path = f"/proc/{pid_dir}/status"
        try:
            name = ""
            rss_kb = 0
            with open(status_path, "r") as f:
                for line in f:
                    if line.startswith("Name:"):
                        name = line.split(":", 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss_kb = int(line.split(":", 1)[1].strip().split()[0])
                        break  # VmRSS comes after Name; we have both now
            if rss_kb > 0:
                procs.append((int(pid_dir), name, rss_kb / (1024.0 * 1024.0)))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    procs.sort(key=lambda p: p[2], reverse=True)
    return procs[:n]


# ANSI colors for console threshold highlighting
COLOR_RESET = "\033[0m"
COLOR_WARN = "\033[33m"   # yellow
COLOR_CRIT = "\033[31m"   # red
COLOR_OK = "\033[32m"     # green


def level_for(pct: float, warn: float, crit: float) -> Tuple[str, str]:
    """Return (label, ansi_color) for a usage percentage."""
    if pct >= crit:
        return "CRIT", COLOR_CRIT
    if pct >= warn:
        return "WARN", COLOR_WARN
    return "OK", COLOR_OK


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Host RAM/swap monitor (OOM watchdog) for pipeline load testing."
    )
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Sampling interval in seconds (default: 2.0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total run time in seconds; omit to run until Ctrl+C")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path (default: no CSV, console only)")
    parser.add_argument("--top", type=int, default=0,
                        help="Show top-N processes by RSS each sample (0 = off)")
    parser.add_argument("--warn-pct", type=float, default=85.0,
                        help="RAM%% for WARN highlighting (default: 85)")
    parser.add_argument("--crit-pct", type=float, default=93.0,
                        help="RAM%% for CRIT highlighting (default: 93)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors in console output")
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    def paint(text: str, color: str) -> str:
        return f"{color}{text}{COLOR_RESET}" if use_color else text

    # CSV setup
    csv_file = None
    csv_writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(out_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "ram_used_gb", "ram_total_gb", "ram_pct",
            "swap_used_gb", "swap_total_gb", "swap_pct", "top_process",
        ])

    # Peak trackers for the summary
    peak_ram_gb = 0.0
    peak_ram_pct = 0.0
    peak_swap_gb = 0.0
    peak_at = ""
    samples = 0

    # Handle Ctrl+C cleanly to always print the summary
    stop = {"flag": False}

    def handle_sigint(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    print(f"System RAM/swap monitor — interval {args.interval}s, "
          f"WARN {args.warn_pct:.0f}% / CRIT {args.crit_pct:.0f}%"
          + (f", CSV → {args.output}" if args.output else "")
          + "  (Ctrl+C to stop)\n")

    start = time.monotonic()
    try:
        while not stop["flag"]:
            info = read_meminfo()
            u = compute_usage(info)
            now = datetime.now().strftime("%H:%M:%S")

            # Track peaks
            if u["ram_used_gb"] > peak_ram_gb:
                peak_ram_gb = u["ram_used_gb"]
                peak_ram_pct = u["ram_pct"]
                peak_at = now
            peak_swap_gb = max(peak_swap_gb, u["swap_used_gb"])

            label, color = level_for(u["ram_pct"], args.warn_pct, args.crit_pct)

            top_str = ""
            if args.top > 0:
                tops = top_processes(args.top)
                top_str = ", ".join(f"{name}={rss:.1f}G" for _, name, rss in tops)

            line = (
                f"{now}  RAM {u['ram_used_gb']:5.1f}/{u['ram_total_gb']:.1f} GB "
                f"({u['ram_pct']:5.1f}%) [{label}]  "
                f"SWAP {u['swap_used_gb']:4.1f}/{u['swap_total_gb']:.1f} GB "
                f"({u['swap_pct']:4.1f}%)"
            )
            print(paint(line, color) + (f"   top: {top_str}" if top_str else ""))

            if csv_writer:
                csv_writer.writerow([
                    now, f"{u['ram_used_gb']:.3f}", f"{u['ram_total_gb']:.3f}",
                    f"{u['ram_pct']:.2f}", f"{u['swap_used_gb']:.3f}",
                    f"{u['swap_total_gb']:.3f}", f"{u['swap_pct']:.2f}", top_str,
                ])
                csv_file.flush()

            samples += 1

            if args.duration is not None and (time.monotonic() - start) >= args.duration:
                break

            # Sleep in small slices so Ctrl+C is responsive
            slept = 0.0
            while slept < args.interval and not stop["flag"]:
                chunk = min(0.2, args.interval - slept)
                time.sleep(chunk)
                slept += chunk
    finally:
        if csv_file:
            csv_file.close()

    print("\n" + "=" * 60)
    print(f"Summary — {samples} samples over "
          f"{time.monotonic() - start:.0f}s")
    print(f"  Peak RAM : {peak_ram_gb:.1f} GB ({peak_ram_pct:.1f}%) at {peak_at}")
    print(f"  Peak SWAP: {peak_swap_gb:.1f} GB")
    if args.output:
        print(f"  CSV      : {args.output}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

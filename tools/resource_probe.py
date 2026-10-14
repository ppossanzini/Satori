#!/usr/bin/env python3
"""Measure a command's Linux resources and concurrent incremental builds."""
import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def counters(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.replace(":", "").split()
        if len(fields) > 1 and fields[1].isdigit():
            values[fields[0]] = int(fields[1])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--builds", type=int, default=3)
    parser.add_argument("--build-delay", type=float, default=10)
    parser.add_argument("--cancel-after", type=float)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-pid", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.builds < 0 or args.build_delay < 0 or args.timeout <= 0:
        parser.error("builds and build-delay must be nonnegative; timeout must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_command = ["dotnet", "build", "Satori.sln", "-c", "Release", "--no-restore", "-m:1"]
    baseline = []
    if args.builds:
        subprocess.run(build_command, check=True, capture_output=True, text=True)
    for index in range(args.builds):
        start = time.monotonic()
        result = subprocess.run(build_command, capture_output=True, text=True)
        baseline.append({"seconds": time.monotonic() - start, "exit": result.returncode})
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    before = counters("/proc/vmstat")
    samples, builds = [], []
    requested_at = None
    done = threading.Event()
    start = time.monotonic()
    with output.with_suffix(".stdout").open("w") as stdout, output.with_suffix(".stderr").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        sampled_pid = args.sample_pid or process.pid
        print(f"started pid={process.pid}: {' '.join(command[:3])}", flush=True)

        def monitor():
            while not done.is_set():
                try:
                    status = counters(f"/proc/{sampled_pid}/status")
                    vm = counters("/proc/vmstat")
                    cpu_fields = Path(f"/proc/{sampled_pid}/stat").read_text().rsplit(")", 1)[1].split()
                    samples.append({"seconds": time.monotonic() - start,
                                    "rss_kib": status.get("VmRSS", 0),
                                    "hwm_kib": status.get("VmHWM", 0),
                                    "swap_kib": status.get("VmSwap", 0),
                                    "threads": status.get("Threads", 0),
                                    "cpu_ticks": int(cpu_fields[11]) + int(cpu_fields[12]),
                                    "io": counters(f"/proc/{sampled_pid}/io"),
                                    "system_pswpin": vm["pswpin"], "system_pswpout": vm["pswpout"]})
                    if len(samples) % 50 == 0:
                        print(f"elapsed={time.monotonic() - start:.0f}s rss={status.get('VmRSS', 0) / 1024:.0f}MiB", flush=True)
                except (FileNotFoundError, ProcessLookupError):
                    return
                done.wait(0.2)

        def build():
            if done.wait(args.build_delay):
                return
            for index in range(args.builds):
                if process.poll() is not None:
                    return
                tick = time.monotonic()
                result = subprocess.run(build_command, capture_output=True, text=True)
                builds.append({"start_seconds": tick - start, "seconds": time.monotonic() - tick,
                               "exit": result.returncode, "overlapped_entirely": process.poll() is None})

        monitor_thread = threading.Thread(target=monitor)
        build_thread = threading.Thread(target=build)
        monitor_thread.start()
        build_thread.start()
        timed_out = False
        try:
            if args.cancel_after is not None:
                try:
                    process.wait(timeout=args.cancel_after)
                except subprocess.TimeoutExpired:
                    requested_at = time.monotonic()
                    process.send_signal(signal.SIGINT)
            exit_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            exited_at = time.monotonic()
            done.set()
            monitor_thread.join()
            build_thread.join()
        elapsed = exited_at - start
    after = counters("/proc/vmstat")
    report = {"command": command, "exit": exit_code, "seconds": elapsed,
              "sampled_pid": sampled_pid,
              "timed_out": timed_out,
              "cancel_to_exit_seconds": None if requested_at is None else exited_at - requested_at,
              "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
              "baseline_builds": baseline, "concurrent_builds": builds,
              "system_swap_in_pages": after["pswpin"] - before["pswpin"],
              "system_swap_out_pages": after["pswpout"] - before["pswpout"],
              "samples": samples,
              "limits": ["Process RSS excludes shared GPU allocations; no total GPU RAM claim.",
                         "System swap counters include other processes.",
                         "Incremental builds only; IDE latency and browser workload are not controlled."]}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "command")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""DW04 measurement wrapper: run a command, report wall time and child peak RSS.

Peak RSS is the child's ru_maxrss (KiB on Linux), which includes the memory-mapped
model pages that were touched. It is a peak working set, not a resident-set guarantee.
"""
import argparse
import os
import resource
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="label written in the report line")
    parser.add_argument("--out", help="append the report line to this file")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given")

    env = dict(os.environ)
    start = time.perf_counter()
    process = subprocess.Popen(command, env=env)
    pid, status, usage = os.wait4(process.pid, 0)
    wall = time.perf_counter() - start
    if os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
    else:
        code = 128 + os.WTERMSIG(status)
    peak_mib = usage.ru_maxrss / 1024.0
    line = f"{args.label}\twall_s={wall:.2f}\tpeak_rss_mib={peak_mib:.0f}\texit={code}"
    print(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())

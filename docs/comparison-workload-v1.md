# Comparison Workload v1

Status: DW02 proposal, pending DW03 approval. Reproducible concurrent workload for the responsiveness
measurement (protocol section 6). Every value below is fixed before candidate comparison.

## Components

| Component | Fixed configuration |
|---|---|
| IDE | VS Code with the Satori solution folder open, the pinned extensions, one source file active |
| Browser | one browser, a fixed set of tabs loaded from a fixed list; JavaScript enabled |
| Build | `dotnet build Satori.sln -c Release --no-restore -m:1` from the repository root |

The IDE project, the browser tab list and the build command are frozen before P02 and reused unchanged for
every candidate.

## Procedure

1. Baseline without inference: run the workload, record build duration and IDE interaction latency.
2. Run the same workload with inference active (prefill and decode) and record the same metrics.
3. Repeat the sequence three times per candidate; report the individual runs.

## Metrics

| Group | Metric |
|---|---|
| Build | wall time of the build command, per run |
| IDE | latency of a fixed interaction (go-to-definition on a fixed symbol), per run |
| CPU/GPU | process and total CPU load; GPU load where applicable |
| Memory | total runtime memory against the budget |
| Swap | sustained swap activity over the window |
| I/O | SSD read/write contention over the window |

## Observation window

The window covers one full build plus the interaction sample, with a stabilization period before it.
Sustained-swap and I/O-contention rejection thresholds are fixed in DW03 from the baseline runs; they are
not asserted before the baseline exists.

## Reproducibility

Every report records IDE and browser versions, the extension set, the build command, the tab list, thread
limits, context, memory budget and the candidate under test. Regenerable outputs live under `.temp/` or
`reports/`; durable evidence under `docs/benchmarks/<topic>-<date>/`.

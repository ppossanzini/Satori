# Comparison Workload v1

Status: owner-approved Intel workload on 2026-10-14. Build slowdown must be ≤25%.
The primary Geekom A5 validation remains separate.

## Components

| Component | Fixed configuration |
|---|---|
| IDE | Current VS Code instance with the Satori solution open; extensions unchanged during measurement |
| Browser | Current browser instances and tabs kept unchanged during measurement; private URLs are not collected |
| Build | `dotnet build Satori.sln -c Release --no-restore -m:1` from the repository root |

The manual interaction sample is go-to-definition on `DenseSession.Append`.
Report its latency separately; automated build timing does not substitute for this sample.

## Procedure

1. Baseline without inference: warm the incremental build, then record three build durations and IDE interaction latency.
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

The automated window covers the inference command from start to exit, sampled every 200 ms.
The three concurrent builds begin after 10 s and must finish before inference exits.
Report system swap-in/out pages and process swap samples. Zero page activity passes the no-sustained-swap criterion for that window only.
The manual interaction window and SSD contention must also be reported before full acceptance.

## Reproducibility

Every report records IDE and browser versions, the extension set, the build command, the tab list, thread
limits, context, memory budget and the candidate under test. Regenerable outputs live under `.temp/` or
`reports/`; durable evidence under `docs/benchmarks/<topic>-<date>/`.

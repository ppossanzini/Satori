# Three-path planner verification

Eight process cases pass: fresh, cached CPU, wrong identity, missing/invalid resident samples, malformed JSON and synthetic cached MatVec/Resident choices. Synthetic timings only exercise policy branches. All outputs are Once/token 9038. CPU cache hits construct zero candidate matrices; GPU cache hits construct only the selected path, with no calibration dispatches.

Three independent context reports retain hardware/runtime and every sample. Release, no validation/timestamps, excluded prefill, three warmups and seven sequential measured appends at positions 4–10 or 67–73. CPU medians 4.090–5.754 ms, MatVec 142.004–193.679 ms, Resident 60.457–80.819 ms. CPU remains selected. Resident improves on the previous GPU path but does not justify default acceleration. These context results are separate from the short two-token calibration policy and the final complete benchmark.

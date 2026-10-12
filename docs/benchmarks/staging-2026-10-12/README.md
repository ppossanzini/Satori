# Persistent staging mapping

T36 maps each host-visible allocation once; unmap follows pending work completion and precedes memory release. Whole-allocation flush/invalidate is retained. Nine hardware tests pass with SyncVal: changing inputs, zero hot allocations, concurrent disposal and disposed owners. Noncoherent hardware remains unverified.

Three Release processes, timestamps off, three warmups and seven eleven-token samples per process, identical to T32. Decode medians: 152.798 ms, 154.438 ms, 149.226 ms. T32: 149.783–157.967 ms. CPU/GPU numerical gates and zero hot allocations pass. Waits still dominate; no inference speedup established. Raw reports and phase summary are retained.

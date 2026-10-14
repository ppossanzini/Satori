# Memory-Mapped Weights

`MappedWeights` opens an existing file read-only and maps its entire length into the 64-bit process. It does not load the entire contents into a managed array. An empty file is rejected because it is not a mappable region. Opening or mapping errors remain I/O errors; a failed constructor releases resources already acquired.

## Access

`Read<TState>(offset, length, action, state)` exposes a `ReadOnlySpan<byte>` only during the callback. Offsets are 64-bit; each window has an `int` length, as required by spans. Files larger than 2 GiB are read in windows without truncating offsets. Ranges are checked before acquiring the pointer; an empty range at the end of the file is valid.

Generic state also supports ref structs to pass working data without closures or boxing. A reused static callback produces no managed allocations on the access path after warm-up. The owner does not copy contents; a callback may choose to copy them.

## Concurrent Disposal

A short lock makes the state check and pointer acquisition through `SafeMemoryMappedViewHandle.AcquirePointer` atomic. The callback runs outside the lock: multiple readers can proceed concurrently. The pointer is released in `finally`, even if the callback throws an exception.

`Dispose` immediately prevents new accesses and closes the owner. References acquired by readers keep the mapping valid until their callbacks finish; actual unmapping may therefore occur after `Dispose` returns. A callback may close the owner without invalidating its own bytes. Disposal is idempotent.

No public pointer, returned span or lease that the caller must release separately. Byte operations must finish within the synchronous callback. Caller unsafe APIs can bypass this constraint and are outside the contract.

## Limits

The file must remain unchanged throughout the lifetime of the mapping and active accesses. `FileShare.Read` prevents incompatible opens where the system enforces these constraints; it does not guarantee protection from external changes or truncation on all systems. Mapping does not guarantee RAM residency: the operating system manages the page cache and I/O.

This component does not know the weight format, validate tensors or transfer data to the GPU. `MappedTernary2Matrix` adds STQ2 loading and execution through the same CPU kernels as in-memory matrices; the contract is in [ternary2-v1.md](ternary2-v1.md). Current validation was performed on Linux x64.

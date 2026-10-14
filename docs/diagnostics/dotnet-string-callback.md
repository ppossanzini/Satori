# String Callbacks and the .NET JIT

The parser uses GgufStringResult to return strings from a mapped read. The wrapper replaces the single-string array and preserves the protected-read contract. The workaround is limited to the metadata loading path.

Observed environment: Arch Linux x64, .NET 10.0.12, Release build. Two suite runs ended with SIGSEGV in the tiered compiler thread, with AVX2 enabled and disabled. The dump and symbols matching installed binaries identify a null MethodTable in CEEInfo::isExactType, called by Compiler::gtCanSkipCovariantStoreCheck during inlining. The parser string callback returns its value through a covariant array on the affected path.

A diagnostic check with PGO disabled passes. The local fix removes that temporary array. Final checks retain default PGO and tiered compilation: 109 tests passed with AVX2, 109 without AVX2 and one additional standard run. The regression test repeats 1000 opens and reads, checking names and scalar types. Neither runtime configuration nor the project SDK is changed.

[Microsoft documentation](https://learn.microsoft.com/en-us/dotnet/core/runtime-config/compilation) defines the options used only during diagnosis. Symbols were downloaded for the exact build IDs of installed binaries. Temporary dumps, symbols and diagnostic tools are removed; no new runtime package enters the solution. The workaround does not imply that every cause of JIT crashes is resolved for other workloads or versions.

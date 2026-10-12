# Baseline scalar — 12 ottobre 2026

Intel Core Ultra 7 155H, Arch Linux x64, .NET 10.0.12, SDK 10.0.112. Kernel scalar a un thread; BenchmarkDotNet 0.14.0 in-process, tre iterazioni da circa 200 ms dopo warm-up.

| Matrice | Block size | Media ms | Deviazione standard ms | Byte/peso | GB/s effettivi | Allocazioni byte/op |
|---|---:|---:|---:|---:|---:|---:|
| 1024² | 64 | 7.555 | 0.159 | 0.312500 | 0.0434 | 0 |
| 1024² | 128 | 9.151 | 0.536 | 0.281250 | 0.0322 | 0 |
| 1024² | 256 | 11.063 | 1.118 | 0.265625 | 0.0252 | 0 |
| 2048² | 64 | 39.999 | 1.630 | 0.312500 | 0.0328 | 0 |
| 2048² | 128 | 31.004 | 1.107 | 0.281250 | 0.0380 | 0 |
| 2048² | 256 | 30.372 | 0.564 | 0.265625 | 0.0367 | 0 |
| 4096² | 64 | 162.336 | 9.078 | 0.312500 | 0.0323 | 0 |
| 4096² | 128 | 179.409 | 3.867 | 0.281250 | 0.0263 | 0 |
| 4096² | 256 | 161.493 | 7.048 | 0.265625 | 0.0276 | 0 |

36/36 verifiche numeriche superate. Nove misure di prestazioni concluse. I report JSON conservano errori, ambiente e misure complete; `benchmarkdotnet.md` riporta gli intervalli di confidenza.

Questa esecuzione breve presenta dispersione e intervalli di confidenza ampi. La priorità alta richiesta da BenchmarkDotNet non è stata concessa dal sistema; processo a priorità ordinaria. Le differenze fra block size non dimostrano un vantaggio ripetibile. Confrontare scalar e AVX2 nello stesso protocollo, ripetendo le misure su hardware stabile prima di scegliere un kernel.

Le misure includono la decodifica, escludono quantizzazione e setup. La banda effettiva dei pesi non è la banda RAM misurata. Nessuna conclusione su token/s o qualità LLM. Metodo in [benchmark-method.md](../../benchmark-method.md).

# Confronto scalar / AVX2 — 12 ottobre 2026

Intel Core Ultra 7 155H, Arch Linux x64, SDK 10.0.112, runtime .NET 10.0.12. Due esecuzioni separate dello stesso job breve, un thread, stessi dati e stesso protocollo. Ogni esecuzione comprende tre warm-up e tre iterazioni misurate da circa 200 ms.

| Matrice | Blocco | Scalar ms (run 1 / 2) | AVX2 ms (run 1 / 2) | Rapporto scalar/AVX2 (run 1 / 2) |
|---|---:|---:|---:|---:|
| 1024² | 64 | 7.236 / 6.333 | 0.247 / 0.231 | 29.32× / 27.37× |
| 1024² | 128 | 7.211 / 6.333 | 0.239 / 0.225 | 30.17× / 28.12× |
| 1024² | 256 | 7.195 / 7.286 | 0.241 / 0.238 | 29.81× / 30.62× |
| 2048² | 64 | 27.701 / 28.128 | 1.016 / 1.022 | 27.27× / 27.53× |
| 2048² | 128 | 26.948 / 28.794 | 0.896 / 0.978 | 30.07× / 29.45× |
| 2048² | 256 | 27.004 / 28.666 | 0.916 / 0.911 | 29.47× / 31.48× |
| 4096² | 64 | 107.649 / 122.755 | 3.827 / 3.941 | 28.13× / 31.15× |
| 4096² | 128 | 114.150 / 112.365 | 3.602 / 3.706 | 31.69× / 30.32× |
| 4096² | 256 | 106.646 / 118.433 | 3.643 / 3.715 | 29.28× / 31.88× |

Vantaggio osservato: 27.27×–31.88× rispetto al kernel scalar di Satori nelle configurazioni sintetiche misurate. Non è un confronto con llama.cpp né una misura di token/s.

72/72 risultati numerici validi per esecuzione; zero byte gestiti allocati per operazione in tutte le misure. I report BenchmarkDotNet includono dispersione e intervalli di confidenza. Priorità alta negata dal sistema; misure a priorità ordinaria. Il grande vantaggio si ripete in entrambe le esecuzioni, ma non dimostra che AVX2 sia migliore per qualsiasi dimensione, distribuzione o CPU.

Gli snapshot `run1/` e `run2/` conservano ambiente, risultati numerici, prestazioni e report BenchmarkDotNet. Metodo: [benchmark-method.md](../../benchmark-method.md).

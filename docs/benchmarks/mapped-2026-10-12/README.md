# MatVec memory-mapped — 12 ottobre 2026

Intel Core Ultra 7 155H, Arch Linux x64, .NET 10.0.12; un thread, AVX2, blocchi da 64 pesi. BenchmarkDotNet 0.14.0 in-process, tre warm-up e tre iterazioni misurate da circa 200 ms.

| Matrice | Array posseduto ms | Mapping ms | Apertura + validazione ms, 3 prove | Primo calcolo dopo validazione ms, 3 prove |
|---|---:|---:|---|---|
| 1024² | 0.228 | 0.234 | 4.509 / 1.507 / 1.478 | 0.768 / 0.228 / 0.215 |
| 2048² | 0.939 | 0.942 | 6.126 / 5.906 / 5.891 | 1.037 / 0.903 / 0.874 |
| 4096² | 3.760 | 3.751 | 25.744 / 25.954 / 25.988 | 3.837 / 3.731 / 3.785 |

Identità numerica confermata in tutte le nove prove di apertura e nei test critici su tutti i block size. Sei benchmark completati, zero byte gestiti allocati per MatVec. Le misure a regime sono comparabili: questo job breve non dimostra piccoli vantaggi del mapping rispetto agli array.

I file sono appena stati scritti dal setup e possono essere già nella page cache. La validazione iniziale attraversa tutto il payload. Il primo calcolo dopo la validazione non è una misura SSD a freddo; il primo campione può includere inizializzazioni o JIT della callback. Apertura/validazione includono le proprie allocazioni, escluse dal benchmark a regime.

Priorità alta non concessa dal sistema: misure a priorità ordinaria. Report completo con dispersione in `benchmarkdotnet.md`; misure grezze JSON accanto a questo file. Metodo in [benchmark-method.md](../../benchmark-method.md).

# Ternary2 e TernaryBitplane

Confronto CPU su identici codici ternari, scale FP32 e attivazioni. Il bitplane deriva da conversione senza perdita del payload Ternary2; nessuna nuova quantizzazione. La forma è quadrata, 1024/2048/4096; block size 64/128/256. Seed 1729, corpus v1.

Ogni blocco occupa `4 + blockSize/4` byte in entrambi i layout: 0,3125 / 0,28125 / 0,265625 byte per peso nelle forme complete. Non c'è un risparmio RAM del bitplane rispetto a Ternary2.

`scalar/` conserva la misura precedente all'ottimizzazione. `run1/` e `run2/` confrontano tutti e quattro i percorsi nello stesso job: Ternary2 scalar/AVX2 e bitplane scalar/AVX2. `numerical.json` verifica 27 combinazioni critiche e nove matrici grandi: identità scalar, identità degli output SIMD dei due layout e limite FP64 del kernel originale sugli stessi pesi. I test aggiungono dimensioni con code parziali e verifica delle allocazioni.

BenchmarkDotNet ShortRun, Release, in-process, singolo thread, tre warmup e tre iterazioni da circa 200 ms. Preparazione, conversione e validazione escluse. Decodifica compresa nella MatVec. Pesi già in memoria; nessuna misura cold SSD o di inferenza. Ambiente, dispersione e intervalli di confidenza nei report completi. Priorità ordinaria.

## Riproduzione

Dalla radice del progetto:

```sh
dotnet build Satori.sln -c Release -m:1
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- --layouts --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release --no-build -- --layouts --filter '*'
```

I report rigenerabili vengono scritti in `reports/layouts/`. Il kernel bitplane esplicito AVX2 richiede la relativa capacità; `Execute` fornisce il fallback. Il layout non sostituisce automaticamente Ternary2 nel resto del runtime.

## Esito

72 misure BDN complessive nelle due esecuzioni finali, tutte con zero byte allocati per operazione. Bitplane AVX2 è 20,2–39,2× più veloce del proprio scalar nelle configurazioni misurate: il dispatch automatico del bitplane usa quindi AVX2 quando disponibile.

Rispetto al Ternary2 AVX2, il rapporto di latenza bitplane/Ternary2 varia da 0,64 a 1,54; bitplane è più veloce in 7 delle 18 coppie. La variabilità e gli intervalli ampi del job breve non dimostrano un vantaggio generale del layout. Ternary2 rimane il percorso già adottato; bitplane resta un'alternativa sperimentale da misurare nel contesto d'uso. Nessun planner fra layout viene introdotto. Valori puntuali in `comparison.json`.

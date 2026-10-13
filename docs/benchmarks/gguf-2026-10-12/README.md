# Q8_0 e Q4_0 su pesi reali

The filenames below identify historical verification artifacts only; they do not define supported product models or the engine target.

Corpus: [ggml-org/tiny-llamas](https://huggingface.co/ggml-org/tiny-llamas/tree/99dd1a73db5a37100bd4ae633f4cfce6560e1567), revisione `99dd1a73db5a37100bd4ae633f4cfce6560e1567`. File pubblici scaricati soltanto nello spazio temporaneo, non distribuiti nel repository.

| File | Byte | SHA-256 |
|---|---:|---|
| stories15M-q8_0.gguf | 26671328 | 2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6 |
| stories15M-q4_0.gguf | 19077344 | 6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04 |

Il reader ufficiale gguf-py, revisione `efe3626b119ff81033e309c3ba07a770bef2c546`, produce un indice di 57 tensori per file. Satori confronta nomi, forme, tipi, offset assoluti e lunghezze con quell'indice. Le quattro piccole golden del repository sono state lette anche dal reader ufficiale.

Le reference `q*-reference.json` contengono il catalogo e prodotti FP64 per tre matrici: attention Q 288×288, FFN gate 768×288 e FFN down 288×768 (righe×colonne). Decodifica indipendente NumPy da byte, input `((i*17)%31-15)/16`; scala FP16 e posizione dei codici seguono la specifica fissata. Nessun codec Satori è usato per generare le attese.

La tolleranza del kernel è `gamma(n) × sum(abs(w*x)) + n × float.Epsilon`, con `n = columns + columns/32 + 2`, `gamma(n)=n*2^-24/(1-n*2^-24)`. I report scalar/auto registrano massimo errore diviso per il limite e zero allocazioni. Le loro latenze sono sonde diagnostiche; il confronto prestazionale usa i report BenchmarkDotNet nelle sottocartelle.

Benchmark: Release, processo a 64 bit, singolo thread, mapping e validazione esclusi dal tempo, pagine già toccate. Stesse forme e attivazioni tra i due formati, pesi provenienti dai rispettivi file quantizzati. Q4_0 e Q8_0 non rappresentano pesi identici: il confronto misura kernel/formato, non qualità di un modello. ShortRun in-process: tre warmup, tre iterazioni da circa 200 ms. Nessuna pretesa su cold SSD, token/s o qualità linguistica. Ambiente completo e intervalli di confidenza nei report BDN; priorità ordinaria, elevazione della priorità non disponibile.

## Riproduzione

Dalla radice del progetto:

```sh
python3 tools/download_gguf_corpus.py
dotnet run --project validation/Satori.Gguf.Validation -c Release -- .temp/gguf-corpus/stories15M-q8_0.gguf docs/benchmarks/gguf-2026-10-12/q8-reference.json
dotnet run --project validation/Satori.Gguf.Validation -c Release -- .temp/gguf-corpus/stories15M-q4_0.gguf docs/benchmarks/gguf-2026-10-12/q4-reference.json
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --gguf .temp/gguf-corpus --filter '*'
```

Per rigenerare le reference, fornire separatamente il reader ufficiale fissato e NumPy:

```sh
PYTHONPATH=<checkout-ufficiale>/gguf-py python3 tools/gguf_reference.py .temp/gguf-corpus/stories15M-q8_0.gguf .temp/gguf-corpus/q8-reference.json
```

GGML/llama.cpp e NumPy servono soltanto alla generazione esterna delle reference. Non sono dipendenze di build o runtime di Satori.

## Esito AVX2

Due esecuzioni, sei coppie scalar/AVX2 per esecuzione. Speedup osservato: Q8_0 4,47–5,15×; Q4_0 4,62–8,99×. Tutte le misure riportano zero byte allocati per operazione. `comparison.json` lega ciascun rapporto al formato, tensore e run; i report BDN completi conservano dispersione e ambiente. Il kernel automatico può quindi usare AVX2, mantenendo scalar sulle CPU che non lo supportano.

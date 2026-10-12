# Satori

Motore di calcolo quantizzato C#/.NET per inferenza locale, con particolare attenzione ai modelli piccoli e alle quantizzazioni low-bit.

## Target e ordine di sviluppo

- CPU x64 Intel/AMD: kernel scalar e SIMD, primo target.
- Vulkan Compute: primo backend GPU previsto, dopo la validazione dei kernel CPU.
- Metal: memorandum per un futuro backend su macOS e Apple Silicon.
- ROCm: memorandum per un futuro backend su GPU AMD.

Metal e ROCm sono target futuri da valutare; non rientrano nella prima milestone.

## Avvio

Richiede SDK .NET 10.0.112 (o patch compatibile), C# 14. Baseline iniziale verificata su Linux x64.

```sh
dotnet restore Satori.sln
dotnet build Satori.sln -c Release --no-restore -m:1
dotnet test Satori.sln -c Release --no-build --no-restore -m:1
dotnet run --project src/Satori.Console -c Release --no-build
```

La console quantizza una matrice FP32 2×3 in Ternary2 ed esegue la MatVec direttamente sui blocchi compressi. Stampa `[4, -2.6666667]` e 40 byte occupati, incluse scale e padding. Su matrici così piccole il padding prevale sulla compressione.

Ternary2 supporta blocchi da 64, 128 o 256 pesi, scale FP32, importazione validata del payload e ricostruzione esplicita dei pesi per analisi numerica. La specifica binaria è in [docs/ternary2-v1.md](docs/ternary2-v1.md). Il kernel non espande la matrice e non alloca memoria gestita durante il calcolo.

`Ternary2MatVec.Execute` seleziona AVX2 quando disponibile e scalar altrimenti. `ExecuteScalar` permette di usare esplicitamente la reference; `ExecuteAvx2` include lo stesso fallback per CPU senza supporto. Le code parziali vengono elaborate scalar senza letture oltre la fine dell'input.

I test sono riservati ai passaggi critici: formato binario, corruzione dei dati, estremi numerici, correttezza dei kernel e memoria. Bootstrap e passaggi semplici si verificano con build e demo.

## Pesi memory-mapped

`MappedWeights` apre i pesi in sola lettura e permette accessi senza copia tramite callback, con offset a 64 bit. La chiusura impedisce nuovi accessi e mantiene validi quelli già avviati fino al termine delle callback. `MappedTernary2Matrix` carica un file STQ2 validato ed esegue MatVec scalar o AVX2 direttamente sui byte mappati. Contratto e limiti in [docs/mapped-weights.md](docs/mapped-weights.md).

Demo di salvataggio e caricamento, usando un percorso nuovo:

```sh
dotnet run --project src/Satori.Console -c Release -- save demo.stq2
dotnet run --project src/Satori.Console -c Release -- run demo.stq2
```

La demo stampa `[4, -2.6666667]`. Il comando `run` usa `[2, -1, 3]` per matrici con tre colonne e un vettore di uno negli altri casi, mostrando al massimo otto risultati. È una console di laboratorio, non un generatore di testo. `Save` non sovrascrive file esistenti. Il contenitore STQ2 v1 è descritto nella [specifica Ternary2](docs/ternary2-v1.md); il payload di ogni matrice è limitato a `int.MaxValue` byte.

Il piano di sviluppo è in [TASKS.md](TASKS.md).

## Corpus numerico e benchmark

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --filter '*'
```

Il primo comando verifica il corpus con entrambi i kernel e salva gli errori numerici; il secondo confronta scalar e AVX2 con BenchmarkDotNet per matrici 1024², 2048² e 4096², con blocchi da 64/128/256 pesi. I report rigenerabili sono in `reports/cpu/`: ambiente, errori numerici, prestazioni JSON e report BenchmarkDotNet con dispersione statistica. Su hardware senza AVX2 il secondo metodo usa il fallback scalar; il report ambiente indica la capacità effettiva.

Le misure includono la decodifica dei pesi ma escludono quantizzazione e preparazione. Sono eseguite a caldo, su un thread e nello stesso processo; la banda effettiva conta soltanto il payload quantizzato, scale e padding inclusi. Non rappresenta la banda fisica della RAM. Il job breve serve come baseline di sviluppo; per scegliere ottimizzazioni servono confronti ripetuti sullo stesso hardware.

Prima esecuzione salvata: [baseline scalar del 12 ottobre 2026](docs/benchmarks/scalar-2026-10-12/README.md). Protocollo ed errori: [metodo del laboratorio](docs/benchmark-method.md).

Confronto ripetuto: [scalar / AVX2, due esecuzioni](docs/benchmarks/avx2-2026-10-12/README.md). Su Intel Core Ultra 7 155H, AVX2 ha mostrato un vantaggio di circa 27–32× sul nostro kernel scalar nelle configurazioni sintetiche misurate. Non è una misura di inferenza LLM o un confronto con altri runtime.

Per confrontare memoria posseduta e mapping:

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --mapped --filter '*'
```

Report in `reports/mapped/`: apertura/validazione e primo calcolo in `startup.json`, regime in `performance.json`. Il primo calcolo avviene dopo la validazione delle pagine e non rappresenta una lettura SSD a freddo. [Prima esecuzione mapped](docs/benchmarks/mapped-2026-10-12/README.md).

## GGUF

Il [profilo GGUF v1](docs/gguf-subset-v1.md) è implementato: contenitori v3 little-endian, catalogo F32/F16/Q4_0/Q8_0, metadati scalar/array e accesso ai tensori per finestre mappate. `GgufQuantizedMatrix` valida le scale del tensore Q8_0/Q4_0 selezionato e offre MatVec scalar e AVX2 con fallback. F32/F16 sono catalogati; tokenizer e inferenza non sono ancora implementati.

Le [fixture critiche](docs/gguf-fixtures-v1.md) verificano layout e rifiuto dei file corrotti. Il [corpus pubblico e i report](docs/benchmarks/gguf-2026-10-12/README.md) confrontano cataloghi col reader ufficiale e calcoli con reference FP64 indipendenti. Riproduzione:

```sh
python3 tools/download_gguf_corpus.py
dotnet run --project validation/Satori.Gguf.Validation -c Release -- .temp/gguf-corpus/stories15M-q8_0.gguf docs/benchmarks/gguf-2026-10-12/q8-reference.json
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --gguf .temp/gguf-corpus --filter '*'
```

## Layout ternario bitplane

`TernaryBitplaneMatrix.FromTernary2` converte senza perdita, mantenendo le scale FP32 e separando i bit positivi e negativi. `ToTernary2` riproduce il payload originale byte per byte. Il nuovo [formato STB1](docs/ternary-bitplane-v1.md) supporta import validato e salvataggio/caricamento; occupa gli stessi byte di Ternary2.

`TernaryBitplaneMatVec.Execute` usa AVX2 dove disponibile e scalar altrimenti. Il confronto con Ternary2 usa identici pesi, scale e input; include tutta la decodifica:

```sh
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --layouts --validate
dotnet run --project benchmarks/Satori.Benchmarks -c Release -- --layouts --filter '*'
```

Report e limiti del confronto: [benchmark dei layout](docs/benchmarks/bitplane-2026-10-12/README.md). La suite critica si può eseguire anche con `DOTNET_EnableAVX2=0` per provare il fallback reale. Per Q8_0/Q4_0 e bitplane, `ExecuteAvx2` esplicito richiede AVX2; usare `Execute` per il dispatch portabile.

## Primo modello selezionato

[Stories15M Q8_0, profilo v1](docs/stories15m-profile-v1.md): modello base TinyStories, 6 layer, hidden 288, contesto 128. Tokenizer e operatori sono fissati; il forward Satori resta da implementare. Le [fixture indipendenti](fixtures/models/stories15m-q8-v1/README.md) includono token, logits completi, stati intermedi e sequenze greedy, confrontati tra FP64 e runtime ufficiale. Il confronto usa gli stessi pesi ricostruiti per distinguere errori di calcolo dalla quantizzazione delle attivazioni del riferimento.

## Operatori del primo modello

Implementati RMSNorm, RoPE, attention causale, SwiGLU, somma residuale, lettura delle norme F32 e gather delle embedding Q8_0 per Stories15M. Operatori verificati singolarmente contro NumPy FP64, senza allocazioni durante il calcolo. Contratto e riproduzione in [docs/stories15m-operators.md](docs/stories15m-operators.md). Tokenizer Stories15M implementato e verificato sui sedici casi ufficiali, con decodifica UTF-8 in streaming e fixture offline. [Contratto tokenizer](docs/stories15m-tokenizer.md). `StoriesModel.Forward` ricalcola un prefisso fino a 128 token e restituisce i logits dell’ultimo token dai pesi Q8_0 mappati. [Contratto e verifica numerica](docs/stories15m-forward.md). `StoriesModel.CreateSession` fornisce la cache incrementale FP32: 128 token, reset e zero allocazioni a regime, con logits identici al ricalcolo completo. [Contratto cache](docs/stories15m-cache.md). Generazione greedy disponibile dalla console: prompt, cache, continuazione UTF-8 e arresto per EOS/budget/contesto. [Avvio e contratto](docs/stories15m-generation.md).

I test `Category=ModelIntegration` richiedono il corpus reale: `python tools/download_gguf_corpus.py`. Per la suite offline senza pesi usare `--filter 'Category!=ModelIntegration'`.

## Generare testo

```sh
dotnet run --project src/Satori.Console -c Release --no-build -- generate .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12
```

Restituisce `, there was a little girl named Lily. She loved`. Stampa solo la continuazione; conteggi e motivo di arresto sono su stderr. Default 32 nuovi token, contesto massimo 128 incluso BOS. Modello richiesto: artefatto Stories15M Q8_0 fissato. Sampling disponibile con `--temperature 0.8 --top-k 40 --top-p 0.9 --seed 123`; temperature zero mantiene greedy. [Regole e API](docs/stories15m-generation.md#sampling-configurabile). Benchmark end-to-end previsto in T26.

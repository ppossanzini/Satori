# Fixture Stories15M Q8_0 v1

Contratto del modello: [profilo v1](../../../docs/stories15m-profile-v1.md). Provenienza, ambiente, sorgenti dei generatori e SHA-256 di ogni file: [manifest.json](manifest.json).

## Contenuto

| Directory | Reference | Uso |
|---|---|---|
| `numpy/` | Forward FP64 indipendente, stessi pesi Q8_0 ricostruiti | Riferimento primario per Satori FP32 senza quantizzazione delle attivazioni |
| `official-f32/` | llama.cpp sul GGUF F32 derivato dagli esatti pesi Q8_0 | Controprova indipendente di formule, ordine dei pesi e RoPE |
| `official-q8/` | llama.cpp sul file Q8_0 originale | Confronto del runtime con il suo percorso quantizzato |

llama.cpp e gguf-py sono fissati alla revisione nel manifest. NumPy 2.5.4. CPU, un thread, cache K/V F32, contesto 128, nessuna GPU, flash attention esclusa. Nessun codice o kernel Satori è eseguito per generare le fixture. Nessuna dipendenza da llama.cpp nel runtime Satori.

Ogni directory contiene tre casi: BOS soltanto (`empty`), `Once upon a time` (`story`) e `Ciao, perché il cielo è blu?` (`italian`). Il JSON conserva gli ID di input, i file dei logits e le sequenze greedy successive. I 15 vettori intermedi rappresentano l'ultimo token: otto punti del primo layer, output di tutti i sei layer, normalizzazione finale.

File numerici senza header: IEEE-754 little-endian, float64 `.f64le` per NumPy e float32 `.f32le` per gli oracoli ufficiali. I logits contengono 32000 elementi in ordine di token ID. Ogni vettore intermedio contiene 288 elementi, ordine delle teste contiguo per Q/K/V. Dimensioni, nomi e hash sono dichiarati nei JSON; lunghezze diverse sono corruzione della fixture.

I JSON ufficiali contengono anche 16 casi tokenizer: otto input, ciascuno con/senza special. I byte UTF-8 di input, pezzi e decodifica sono conservati in esadecimale; NUL e pezzi UTF-8 parziali non dipendono dall'encoding del lettore JSON. I flag di decodifica e la gestione del prefisso sono fissati nel profilo.

## Verifica degli oracoli

- Tutti i valori numerici sono finiti.
- Tokenizer identico nei due file ufficiali.
- Logits ufficiali F32 contro FP64: errore massimo <1e-5 sui tre prompt.
- Vettori intermedi ufficiali F32 contro FP64: errore massimo <2,2e-5.
- Percorso ufficiale Q8_0: differenze sui logits fino a circa 0,305; non usato per fissare la tolleranza del forward con attivazioni FP32.
- Tre sequenze greedy identiche nei tre oracoli; minimi margini fra primo e secondo token registrati.
- Seconda generazione dei tre gruppi identica byte per byte sullo stesso ambiente.

Questi risultati validano le fixture. Tokenizer e forward Satori non sono ancora implementati né confrontati con esse.

## Rigenerazione

Richiede CMake, compilatore C++17 e NumPy. Download e checkout esterno restano nello spazio temporaneo del progetto. Dalla radice:

```sh
python3 tools/download_gguf_corpus.py
mkdir -p .temp/t19
git clone https://github.com/ggml-org/llama.cpp.git .temp/t19/llama.cpp
git -C .temp/t19/llama.cpp checkout efe3626b119ff81033e309c3ba07a770bef2c546
cmake -S tools/model-reference -B .temp/t19/build -DSATORI_LLAMA_SOURCE="$PWD/.temp/t19/llama.cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build .temp/t19/build --target satori-model-reference -j4
.temp/t19/build/satori-model-reference .temp/gguf-corpus/stories15M-q8_0.gguf fixtures/models/stories15m-q8-v1/official-q8
OPENBLAS_NUM_THREADS=1 PYTHONPATH=.temp/t19/llama.cpp/gguf-py python3 tools/stories15m_reference.py .temp/gguf-corpus/stories15M-q8_0.gguf fixtures/models/stories15m-q8-v1/official-q8/reference.json fixtures/models/stories15m-q8-v1/numpy .temp/t19/stories15M-dequantized-f32.gguf
.temp/t19/build/satori-model-reference .temp/t19/stories15M-dequantized-f32.gguf fixtures/models/stories15m-q8-v1/official-f32
python3 tools/model-reference/verify.py
```

La rigenerazione sovrascrive le fixture nelle tre directory. Il manifest registra l'ambiente; risultati FP32 di altri compilatori/hardware si confrontano numericamente, senza assumere hash identici. Il modello F32 derivato (circa 94 MiB) è un artefatto temporaneo di verifica e non viene distribuito.

## Provenienza

Pesi originali di Andrej Karpathy, modello TinyStories15M per llama2.c. [Scheda originale e dichiarazione MIT](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/README.md). [GGUF usato](https://huggingface.co/ggml-org/tiny-llamas/blob/99dd1a73db5a37100bd4ae633f4cfce6560e1567/stories15M-q8_0.gguf). [Runtime ufficiale di riferimento](https://github.com/ggml-org/llama.cpp/tree/efe3626b119ff81033e309c3ba07a770bef2c546).

La fixture `tokenizer.gguf` conserva soltanto i metadata tokenizer originali e un tensore nullo. Provenienza e hash in `tokenizer-manifest.json`; contratto in [docs/stories15m-tokenizer.md](../../../docs/stories15m-tokenizer.md).

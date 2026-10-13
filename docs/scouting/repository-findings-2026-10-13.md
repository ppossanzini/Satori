# Audit DS4 e AirLLM

Sorgenti letti, non compilati o eseguiti. Le revisioni identificano lo snapshot verificato; i link sotto sono permanenti. Nessun risultato prestazionale upstream è una misura di Satori o del mini PC da 16 GB.

## DS4 small

Snapshot: `fc80bd695da76ee14bcb86820bf900a5cdfda806`. C con backend CUDA/HIP e Metal; licenza principale MIT, con attribuzioni ds4/ggml/DeepSeek. Alcuni componenti hanno licenze proprie da conservare se riutilizzati. Il nome “small” non significa che i modelli descritti entrino in 16 GB.

Nel [kernel CPU](https://github.com/ppossanzini/ds4_small/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4.c#L3829), le attivazioni sono quantizzate una volta in Q8_K e riutilizzate nei dot Q2_K/IQ2_XXS. Le somme intere sono corrette con scale e minimi; sono presenti specializzazioni NEON. Non emerge un kernel AVX2 pronto da quel percorso: Satori deve preservare e misurare le proprie specializzazioni x64.

L'[adattatore CUDA](https://github.com/ppossanzini/ds4_small/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/cuda/mmq/ds4_mmq.cu#L376) seleziona percorsi per formato, hardware e numero di token. Il GEMV Q8 impiega caricamenti vettoriali, staging shared, attivazioni in registri e dot interi `dp4a`; tiene in anticipo la riga successiva. Scratch Q8 persistente e riuso delle attivazioni già quantizzate evitano lavoro ripetuto. Prefill GEMM e decode GEMV richiedono scelte diverse.

La [provenienza dei kernel](https://github.com/ppossanzini/ds4_small/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/cuda/mmq/VENDOR.md) dichiara codice ggml-cuda vendorizzato da llama.cpp `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`, con adattatore e modifiche locali. Conviene studiare anche quell'upstream, conservando attribuzioni e verifiche di parità. Non è un nuovo algoritmo universale trasferibile sostituendo CUDA con Vulkan.

Lo [streaming](https://github.com/ppossanzini/ds4_small/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/SSD_STREAMING.md) riguarda cache limitate di esperti instradati; pesi non instradati, KV e scratch restano nel budget. Il codice di [upload/cache](https://github.com/ppossanzini/ds4_small/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_cuda.cu#L28955) sincronizza il batch e impedisce che copie fallite diventino cache hit. I benchmark documentati su macchine da 128 GB non giustificano capacità o velocità identiche con 16 GB.

Idee trasferibili: selezione GEMM/GEMV per forma, letture coalescenti, riuso dell'input tra proiezioni, scratch limitato, misure per operatore e residency per esperto. Dot interi/cooperative matrix su Vulkan richiedono capability gate e fallback. Quantizzare le attivazioni modifica il percorso numerico: riferimento delle operazioni e qualità del modello vanno verificati separatamente; non basta confrontare il packing dei pesi.

## AirLLM

Snapshot: `7bb8a8600937ff15b81016f286142bcdac7ee2d2`, Apache-2.0. Il [runtime base](https://github.com/lyogavin/airllm/blob/7bb8a8600937ff15b81016f286142bcdac7ee2d2/air_llm/airllm/airllm_base.py) installa hook per caricare i moduli prima del forward e scaricarli dopo, usando Transformers/PyTorch; prefetch con un worker e copie host pinned entro un limite. Lo stesso snapshot disabilita il prefetch quando è richiesta la compressione opzionale. Non assumere che compressione e prefetch siano sempre cumulabili.

Le [utility](https://github.com/lyogavin/airllm/blob/7bb8a8600937ff15b81016f286142bcdac7ee2d2/air_llm/airllm/utils.py) usano bitsandbytes, inclusi percorsi NF4 e decompressione; il [persister](https://github.com/lyogavin/airllm/blob/7bb8a8600937ff15b81016f286142bcdac7ee2d2/air_llm/airllm/persist/safetensor_model_persister.py) salva e legge shard per layer. La numerica è delegata alle librerie e ai modelli sottostanti. Questo ramo non fornisce kernel Vulkan Q4_0 riutilizzabili direttamente.

Idee trasferibili: indice per layer/tensore, prefetch limitato, lifecycle esplicito, distinzione tra moduli residenti e streamed. Il vantaggio principale è capacità con poca VRAM; spazio SSD, RAM host, eventuali layer espansi, KV e latenza restano costi reali. Non adottare PyTorch come dipendenza obbligatoria del runtime .NET.

## AirLLM-plus

Snapshot: `2a9f04035d80abf43fe7579673d08947c49080f1`. Il [README](https://github.com/howardleegeek/airllm-plus/blob/2a9f04035d80abf43fe7579673d08947c49080f1/README.md) dichiara Apache-2.0, ma l'albero letto non contiene un file LICENSE: prima di vendorizzare serve chiarirne il materiale di licenza. L'idea di superblocchi, accesso sequenziale, prefetch e LRU è utile per uno spike I/O.

L'[executor](https://github.com/howardleegeek/airllm-plus/blob/2a9f04035d80abf43fe7579673d08947c49080f1/runtime/executor.py) simula il calcolo con `time.sleep` se manca la callback; `io_throughput_gbps` restituisce un rapporto fra tempi, non GB/s. Non dimostra inferenza LLM né kernel più veloci. La [cache](https://github.com/howardleegeek/airllm-plus/blob/2a9f04035d80abf43fe7579673d08947c49080f1/runtime/cache.py) è CPU-side nello snapshot.

Il [prefetcher](https://github.com/howardleegeek/airllm-plus/blob/2a9f04035d80abf43fe7579673d08947c49080f1/runtime/prefetch.py) scarta il risultato se la coda è piena e stampa errori I/O senza restituirli al consumatore; `AsyncPrefetcher` non è implementato. La consegna non verifica nell'executor l'identità del blocco ricevuto. Sono limiti osservati staticamente, non bug riprodotti eseguendo il repository. Un'implementazione Satori deve avere backpressure, associazione richiesta/risultato, propagazione errori e teardown deterministico.

## Quando usare streaming

Per un denso, ogni token attraversa tutti i layer. Senza riuso in RAM, il limite ideale è `tempo/token >= byte letti da SSD/token ÷ banda SSD effettiva`. Con 4,43 GB riletti e 3 GB/s ipotetici: almeno 1,48 s/token, prima di trasferimenti e calcolo. È un esempio teorico, non un benchmark. Cache OS e sovrapposizione cambiano il caso reale; se tutti i pesi sono già in RAM, non è streaming SSD per token.

Partire da modelli residenti 1,5–7B; valutare 14B con budget e contesto limitati. A 27B Q4_0, i soli blocchi teorici sono circa 15,2 GB, prima di altri tensori/OS/KV: margine insufficiente per assumerlo baseline su 16 GB. MoE riduce gli esperti attivi per token, non il peso totale da archiviare; la convenienza dipende dai miss. L'obiettivo futuro è residency adattiva misurata, non SSD obbligatorio per ogni modello.

# Idee applicabili a Satori

Target: runtime .NET multi-modello, CPU Intel/AMD, Vulkan, 16 GB RAM e pesi compressi. Metal e ROCm restano estensioni future. Streaming e cache sono parte dello scouting per ampliare i modelli eseguibili oltre il working set residente. Le valutazioni sono proposte; nessun percorso nuovo è implementato o misurato sul target.

## Matrice di applicabilità

| Origine / idea | Valutazione | Applicazione a Satori | Limite da verificare |
|---|---|---|---|
| AirLLM: caricare il layer prima del calcolo e rilasciarlo dopo | Applicabile con adattamento | Scheduler .NET con budget e lifetime espliciti | RAM totale: layer corrente, successivo, stato e scratch |
| AirLLM: shard e indice dei pesi per layer | Applicabile come indice | Usare offset GGUF e mapping; shard solo se misure I/O lo richiedono | Evitare ricopia preventiva obbligatoria di tutto il modello |
| AirLLM: prefetch del prossimo layer | Applicabile con adattamento | Lettura asincrona e doppio buffer entro budget | Coprire I/O con calcolo reale; compressione/prefetch non cumulabili nello snapshot upstream |
| AirLLM: offload host→GPU e memoria pinned limitata | Condizionato al backend | Copie e residency esplicite su GPU discreta | Su iGPU host/device condividono RAM: offload non crea capacità aggiuntiva |
| AirLLM: runtime Transformers/PyTorch e bitsandbytes | Non adottare come dipendenza obbligatoria | Studiare il lifecycle; mantenere codec e runtime indipendenti | NF4 non è Q4_0, K, IQ o ternary |
| DS4: cache limitata degli esperti MoE | Applicabile con adattamento | Tenere esperti caldi, caricare solo quelli selezionati mancanti | Miss, granularità, budget e riuso tra token |
| DS4: upload validato prima di segnare una cache hit | Applicabile | Stati richiesto→caricamento→pronto; errore e cancellazione espliciti | Nessun peso parziale visibile al calcolo |
| DS4: quantizzare l'attivazione una volta e riusarla | Applicabile ai kernel compatibili | Condividere input quantizzato tra proiezioni | Riferimento numerico e qualità del modello separati |
| DS4: dispatch per forma, formato e hardware | Applicabile | Separare prefill GEMM e decode GEMV | Soglie derivate da misure sul target |
| DS4: scratch persistente, staging e letture vettoriali | Applicabile con adattamento | Buffer limitati e accessi compatti CPU/Vulkan | Occupancy, banda e memoria condivisa |
| DS4: CUDA dp4a / kernel NEON | Non trasferibili direttamente | Portare l'algoritmo verso AVX2/Vulkan | Capability gate, fallback e parità numerica |
| AirLLM-plus: superblocchi e letture sequenziali | Applicabile come esperimento I/O | Raggruppare range contigui utili | Overfetch: blocchi grandi possono leggere esperti inutilizzati |
| AirLLM-plus: LRU e code prefetch | Applicabile come idea | Cache con limite in byte e backpressure | Non scartare risultati richiesti; associare richiesta e risultato |
| AirLLM-plus: executor e throughput dichiarato | Non usare come prova prestazionale | Sostituire compute simulato con inferenza reale | `time.sleep` non misura kernel; rapporto tempi non è GB/s |
| Prism/Bonsai: codec ternary e trasformazione attivazioni | Applicabile come percorso distinto | PTQ1_0/PQ2_0, Hadamard e operatori ibridi | Non equivalenti a STQ2/STB1 o BitNet I2_S |

Fonti e revisioni DS4/AirLLM: [audit dei sorgenti](repository-findings-2026-10-13.md). AirLLM-plus richiede chiarimento della licenza prima di vendorizzare codice: README Apache-2.0, LICENSE assente nello snapshot letto. DS4 include componenti con proprie attribuzioni. Il riuso delle idee non autorizza a eliminare gli obblighi del codice importato.

## Memoria: tre casi distinti

| Percorso | Cosa resta in memoria | Utilità | Costo |
|---|---|---|---|
| Residente | Pesi compressi completi e stato del modello | Quando tutto entra nel budget | Nessun caricamento SSD ricorrente dei pesi |
| Layer streaming | Pesi indispensabili e finestre di layer | Densi o parti dense che non entrano | Ogni token attraversa tutti i layer; rischio rilettura SSD continua |
| Expert streaming | Pesi condivisi, stato e cache degli esperti | MoE con riuso e working set limitato | I/O dipende dagli esperti selezionati mancanti |

La poca VRAM di AirLLM non equivale a poca RAM totale. Trasferire da RAM a GPU e leggere da SSD a RAM hanno costi differenti. Il mapping sfrutta la cache OS ma non garantisce residency né un limite fisico alla memoria del sistema. Contabilizzare memoria privata, pagine residenti, buffer grafici e cache evitando doppi conteggi sulle iGPU.

Il routing determina gli esperti necessari del layer corrente. Il loro prefetch non è sempre noto in anticipo: usare hit già disponibili, metadati noti e previsioni verificabili. Non assumere che la selezione del token successivo sia prevedibile. Una politica LRU è un punto di partenza, non una prova di efficienza.

## Candidati aggiunti

Dimensioni in GB decimali, verificate tramite API Hugging Face. [Revisioni e hash](apex-bonsai-evidence-2026-10-13.json). Inventario binario GGUF ancora da verificare; nessun peso scaricato.

| Candidato | Pesi testo | Ruolo nello scouting | Gap principale |
|---|---:|---|---|
| [Tiel-Coder 35B-A3B APEX-I-NanoPlus](https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF/tree/ef9ff9001e8205458bf842fd13f43e5318eb2d46) | 12,554 | Caso MoE per residency parziale e cache/streaming; non escluso per RAM | Routing, esperti condivisi, attenzione ibrida/DeltaNet e codec misti |
| [Ternary-Bonsai-2 27B PTQ1_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/b072e1d3b35a0a630cece372c2127528e0994386) | 5,947 | Ternary grande candidato residente; streaming se il budget reale lo richiede | Codec Prism, rotazioni e runtime ibrido |
| Bonsai-2 27B PQ2_0, stesso repository | 7,206 | Confronto packing/velocità con PTQ1_0 | Più traffico pesi, unpacking differente |

APEX è una ricetta mista, non un codec GGUF unico. La scheda dichiara F32, Q8_0, Q3_K, Q4_K, Q6_K, IQ2_S, IQ2_XXS e IQ3_XXS. I circa 3B attivi non rappresentano i pesi totali da archiviare. La scheda contiene valori discordanti di perplexity/BPW e riferimenti di provenienza non uniformi: chiarire base e valutazione prima di assumere qualità equivalente a Q4_K_M. Nessuna promessa di token/s o contesto massimo sul target Satori. Fonte: [scheda APEX](https://huggingface.co/IsValorum/Tiel-Coder-35B-A3B-APEX-I-NanoPlus-GGUF).

Il selettore Bonsai `:TQ1_0` richiesto non trova un file omonimo nella revisione verificata: sono pubblicati PTQ1_0 e PQ2_0. La scheda descrive ternary g128 con scale FP16, rotazione Hadamard a blocchi 1024 e backbone ibrido Qwen3.8. Il fork Prism è il riferimento dichiarato; il supporto upstream non va presunto. I projector visivi sono opzionali per testo: APEX 0,614 GB, Bonsai Q8 0,629 GB. Fonte: [scheda Bonsai](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

Nel [fork Prism verificato](https://github.com/PrismML-Eng/llama.cpp/tree/e8fd209c975c80aa7fb36a8423571ff3b1b7e955), `ggml-common.h` definisce PTQ1_0 come 28 byte/128 pesi e PQ2_0 come 34 byte/128 pesi. `ggml.h` assegna tipi privati 143 e 142. Non rinominare PTQ1_0 in TQ1_0: quest'ultimo usa gruppi diversi. L'albero contiene shader Vulkan di dequantizzazione e MatVec PTQ1_0: sono piste da ispezionare, non supporto completo già validato. Verificare trasformazioni su embedding/output, metadati e fallback, oltre ai dot.

## Cosa non possiamo concludere

- Streaming rende possibile un working set limitato; non garantisce velocità interattiva.
- La compressione dei pesi non riduce automaticamente KV, stato ricorrente o scratch.
- Contesto nominale e benchmark del produttore non provano sostenibilità con 16 GB RAM.
- CUDA/Metal non dimostrano prestazioni Vulkan; presenza di shader non dimostra correttezza end-to-end.
- Non serve un backend chiamato APEX o Bonsai: servono architetture, codec, trasformazioni e politiche di memoria separati.

## Prove che determinano l'applicabilità

1. Inventario GGUF: tipi, forme, offset, tokenizer, architettura, rotazioni, provenienza e componenti opzionali. Fail esplicito per capability mancante.
2. Budget runtime: pesi sempre residenti, cache esperti/layer, stato, scratch e prefetch. Vietare espansione permanente completa in F32.
3. Spike I/O con range rappresentativi: byte SSD/token, hit/miss per esperto, overfetch, tempi lettura/copia e attesa del calcolo. Separare cache fredda e calda.
4. Parità su percorsi critici: decoder/dot, routing top-k, trasformazioni Hadamard, aggiornamento dello stato ibrido e gestione errori/lifetime.
5. Inferenza reale: startup, prefill, TTFT, decode e picco memoria a 2K/4K; qualità coding e italiano con output lunghi. Stabilire soglie accettabili prima di dichiarare il risultato.

Priorità proposta: contratti generici e inventario; codec K/IQ condivisi; prototipo di cache esperti per APEX; percorso ternary/rotazioni Bonsai. Le due esplorazioni possono avanzare senza fissare un solo modello come architettura del prodotto.

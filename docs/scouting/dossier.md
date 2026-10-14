# Scouting: modelli locali e kernel

Current product scope and execution order: [P01–P08](../../TASKS.md#product-execution-plan). Candidate and capability proposals below are scouting evidence; they do not authorize a broader implementation sequence.

Verifica: 13 ottobre 2026. Target: coding locale su CPU Intel/AMD con 16 GB RAM e pesi compressi; Vulkan condizionato alle misure. Le proposte sotto non fissano un nuovo modello come architettura del prodotto. Nessun peso scaricato, codice esterno eseguito o benchmark locale svolto in questo scouting.

## Candidate models

Dimensioni effettive degli artefatti ufficiali Hugging Face, in GB decimali. KV stimata per una sequenza di 2.048 token, F32: `2 × layer × teste KV × dimensione testa × token × 4`. FP16 dimezzerebbe la KV, ma richiede un percorso verificato. File e KV non rappresentano RAM totale: aggiungere OS, tokenizer, attivazioni, scratch e copie del backend. Sulle iGPU RAM e memoria grafica condividono il budget.

| Candidato / fonte ufficiale | Artefatti verificati, GB | KV F32 a 2K | Ruolo proposto |
|---|---:|---:|---|
| [Qwen2.5-Coder 1.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) | Q4_0 1,066; Q8_0 1,895 | 112 MiB | Prima integrazione coding e confronti rapidi Q4/Q8 |
| [Qwen2.5-Coder 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) | Q4_0 4,431; Q8_0 8,099; Q4_K_M 4,683; Q2_K 3,016 | 224 MiB | Primo carico coding rappresentativo; confronto Q4_0/Q4_K_M |
| [Qwen2.5-Coder 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF) | Q4_0 8,518; Q8_0 15,702; Q4_K_M 8,988; Q2_K 5,770 | 768 MiB | Candidato CPU successivo con budget esplicito; Q8 non adatto al budget 16 GB |
| [Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | Q8_0 4,280; Q4_K_M 2,497 | 576 MiB | Seconda architettura e carico generalista; Q8 evita inizialmente nuovi codec |
| [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | Q8_0 8,710; Q4_K_M 5,028 | 576 MiB | Alternativa generalista più pesante, dopo 4B |
| [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | I2_S 1,188 | 300 MiB | Candidato ternary nativo di scouting; non richiesto dalla prima consegna |
| [SmolLM2 1.7B Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | Q4_K_M 1,056 | 768 MiB | Alternativa architettura Llama; il file ufficiale verificato richiede codec K |

Qwen e SmolLM2: Apache-2.0; BitNet: MIT. Non è una classifica di qualità misurata. I modelli coding devono superare test di compilazione/esecuzione su C# e altri linguaggi dichiarati; per uso italiano servono prompt dedicati. Revisioni, byte esatti, SHA degli artefatti quando esposti e topologie sono in [evidence-2026-10-13.json](evidence-2026-10-13.json).

[Riepilogo delle idee applicabili](applicability-summary.md): DS4, AirLLM, AirLLM-plus; candidati APEX e Bonsai, streaming e prove necessarie.

## Compatibility

Il codice corrente espone Q4_0/Q8_0 nei kernel CPU e Q8_0 su Vulkan; Ternary2/STQ2 e TernaryBitplane/STB1 sono formati di laboratorio. Questo non significa che i modelli della tabella siano già eseguibili in Satori. Servono runtime/sessioni derivati dai metadati, tokenizer e chat template, GQA, dimensioni e forme dinamiche, embedding/output anche condivisi e inventario completo dei tipi di ogni GGUF.

Qwen2 richiede anche bias delle proiezioni pertinenti e RoPE/epsilon dal modello. Qwen3 aggiunge normalizzazione Q/K e dimensione testa esplicita: nel 4B `32 × 128 != hidden_size 2560`; ricavare sempre la testa da `hidden/heads` sarebbe errato. [Implementazione Qwen2](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/qwen2/modeling_qwen2.py), [implementazione Qwen3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py), [config Qwen2 7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/raw/main/config.json), [config Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json).

Q2_K permette un candidato coding 14B da 5,77 GB di pesi, ma richiede un nuovo codec/kernel e una prova di qualità: ridurre il file non prova che il coding resti utile. Non è ternary.

[Valutazione Q4_K_M e UD-IQ](quantization-candidates.md): ricette miste, candidati Unsloth, costo dei kernel e prove proposte. NF4/GPTQ/AWQ non sono equivalenti a GGUF Q4_0. Il nome del file non prova che tutti i tensori siano supportati.

BitNet usa W1.58A8, BitLinear, subln, ReLU² e tokenizer Llama 3; il GGUF I2_S non è STQ2/STB1. Vanno rispettate scale e quantizzazione delle attivazioni, non convertiti arbitrariamente i pesi di un modello ordinario in ternary. Riferimenti: [scheda ufficiale](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf), [kernel BitNet](https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-lut.cpp).

## Solution scouting

[Scouting della letteratura sulla quantizzazione, 13 ottobre 2026](quantization-literature-2026-10-13.md): PTQ, kernel CPU LUT, ternary e KV; priorità sperimentali per macchine piccole. Le raccomandazioni non modificano il percorso prodotto selezionato.

Criteri di esclusione: dipendenza obbligatoria CUDA/PyTorch nel runtime .NET; espansione permanente di tutti i pesi; memoria non limitata; risultati con calcolo simulato presentati come inferenza; modello unico incorporato nelle API. Ospitare il calcolo in un servizio esterno non soddisfa il requisito locale indipendente.

| Capacità | Decisione proposta | Motivo / limite |
|---|---|---|
| Eseguire pesi compressi su CPU/Vulkan | Estendere i kernel Satori; studiare DS4 e ggml | Gli algoritmi sono utili; CUDA/HIP/Metal non si importano direttamente in Vulkan |
| Eseguire ternary nativo | Estendere codec/operatori usando BitNet come riferimento indipendente | Il codec custom non rappresenta da solo I2_S e W1.58A8 |
| Tenere pesi e scratch entro il budget | Estendere mapping esistente, residency per tensore/gruppo e contatori | Prima evitare copie/espansioni; streaming solo se il working set non entra |
| Caricare layer/esperti in anticipo | Studiare AirLLM e DS4; implementazione futura condizionata a misure | Obiettivo capacità; nessuna promessa di maggiore velocità |
| Definire cache/prefetch a blocchi | Usare l'idea AirLLM-plus, non il runtime come dipendenza | Prototipo CPU con calcolo simulato, criticità documentate nell'audit |

Ricerca coperta: repository richiesti, ggml/BitNet, asset locali e skill disponibili. Nessun catalogo di librerie interne aziendali è collegato. Le skill aiutano lo scouting e la verifica, non forniscono kernel pronti. [Audit dei sorgenti e idee trasferibili](repository-findings-2026-10-13.md).

## Technical feasibility

| Capacità | Rischio dominante | Stato / prova necessaria |
|---|---|---|
| Interpretare modelli differenti senza API nominali | Tokenizer, GQA, QK norm, forme e tipi misti | Gap noto; dimostrare almeno due topologie con stessi contratti e riferimenti indipendenti |
| Conservare pesi compressi durante il calcolo | Differenze tra codec/scale/attivazioni | Codec base noti; verificare I2_S, K e IQ separatamente |
| Generare codice utile entro 16 GB | Qualità Q4 e memoria di contesto/scratch | Sconosciuto sul target; prova coding + picco RAM e TTFT/decode a 2K/4K |
| Usare SSD quando i pesi non entrano | Miss per token e I/O non nascosto | Sconosciuto sul target; misurare byte letti/token, hit rate e throughput SSD reale |

Capacità decisiva: generare su un modello coding utile entro RAM e latenza interattiva misurate. Nessuna velocità minima è stata ancora approvata; non si può dichiarare raggiunto R-PERF. Conclusione: realizzabile per il perimetro denso 1,5–7B e ternary 2B come ipotesi da validare; 14B condizionato al budget; 27B+ non è una baseline garantita; APEX e Bonsai sono candidati esplorativi con percorsi di memoria e operatori descritti nel riepilogo.

## Validation order

[Il piano prodotto](../../TASKS.md#product-execution-plan) fissa confronto nei runtime di riferimento, selezione di un percorso principale e implementazione delle sole capability necessarie. Gli inventari restano evidenza riutilizzabile.

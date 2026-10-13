# Piano di lavoro per DeepSeek

## Mandato

Realizzare coding locale utile su PC Intel/AMD con 16 GB RAM, mantenendo utilizzabili IDE, browser e compilatore. Selezionare un solo percorso principale dopo il confronto dei candidati nei runtime di riferimento.

Fonti vincolanti: [direzione prodotto](docs/adr/current/product.md), [requisiti e piano P01–P08](TASKS.md#product-execution-plan), [protocollo di validazione](docs/validation-protocol-v1.md). Le macchine di test e i relativi ruoli sono nella sezione 4 del protocollo; il Geekom A5 è il riferimento principale. Il protocollo possiede soglie e misure. Questo documento possiede la sequenza operativa del worker.

## Regole del worker

- Eseguire un task alla volta. Rispettare dipendenze e criteri di completamento.
- Leggere istruzioni locali, stato Git e documenti pertinenti prima delle modifiche. Preservare modifiche preesistenti e cancellazioni intenzionali.
- Riutilizzare descrittori, tokenizer, sessioni, mapping, reader GGUF, kernel e primitive Vulkan presenti solo dopo verifica del loro stato effettivo.
- Le checkbox del precedente piano DS non attestano supporto dei candidati. Verificare codice e prove pertinenti prima di riutilizzare una capacità.
- Nessuna implementazione specifica dei candidati prima della selezione in DW04. Gli strumenti necessari al confronto sono ammessi.
- Runtime esterni ammessi per confronto e oracle indipendenti. Il runtime produttivo Satori resta indipendente da PyTorch, Transformers, llama.cpp e servizi remoti.
- Implementare solo architettura, tokenizer, codec, trasformazioni e politica di memoria richiesti dal percorso selezionato. Dimensioni e stato derivano dai metadati validati.
- Conservare i pesi compressi. Scratch, staging, cache e prefetch hanno limiti in byte. Vietare espansione permanente completa in F32/FP16.
- Verificare licenze, attribuzioni, revisioni e spazio prima di scaricare pesi o importare codice. AirLLM-plus resta fonte di idee finché la licenza non è chiarita.
- Usare `.temp/deepseek/<id>` per materiale temporaneo. Non eliminare corpus preesistenti. Non includere pesi nei commit.
- Test soltanto per invarianti critici: formati, numerica, SIMD/GPU, inferenza, risorse, lifetime e I/O. Documentazione e wiring richiedono controlli mirati.
- Registrare esito, evidenza e limiti in TASK-LOG.md; aggiornare la checkbox; creare un commit del solo task. Non fare push né includere modifiche altrui.
- Se mancano hardware, oracle o decisioni indispensabili, indicare il blocco concreto. Proseguire soltanto le attività indipendenti. Non sostituire misure con stime.
- Aggiornare decisioni correnti e ProjectInfo; non creare registri ADR.

## Verifiche comuni

SDK e linguaggio da `global.json` e progetti. Build: `dotnet build Satori.sln -c Release --no-restore -m:1`; restore solo se necessario. Test numerici nel progetto `tests/Satori.Numerics.Tests`, filtrati sui comportamenti modificati. Test GPU nel progetto `tests/Satori.Vulkan.Tests` con hardware e validation layer disponibili.

Le reference numeriche usano gli stessi pesi e tolleranze motivate per operatore. I confronti tra engine usano lo stesso artefatto, tokenizer, contesto e sampling. Tra modelli o quantizzazioni diverse confrontare il successo dei task, senza imporre identità dei token.

## A — Preparare il confronto (P01)

| Stato / ID | Azione | Dipendenze | Risultato e verifica |
|---|---|---|---|
| [x] DW01 | Verificare asset e capacità riutilizzabili | Nessuna | Inventario conciso di codice, reference e report disponibili. Collegare evidenze per ogni capacità; dichiarare quelle non verificate. Nessun ampliamento del runtime. |
| [x] DW02 | Fissare candidati, runtime e carico concorrente | DW01 | Manifest di Qwen2.5-Coder 7B Q4_K_M, Tiel-Coder APEX-I-NanoPlus e Bonsai-2 27B PTQ1_0: revisioni, hash, licenze, runtime compatibili, hardware e configurazioni secondo la matrice di test del protocollo. Eseguire prima sul riferimento principale; distinguere budget runtime e RAM fisica. Corpus C# congelato, contesti e workload IDE/browser/build riproducibili. Risolvere eventuali discrepanze di provenienza APEX prima delle reference. |
| [x] DW03 | Chiudere le soglie aperte del protocollo | DW02 | Ottenere dal responsabile le decisioni richieste dal protocollo. Proporre valori motivati dalle prove preliminari del carico concorrente; non approvarli autonomamente. Registrare baseline senza inferenza, limiti delle risorse e criteri di accettazione. |

Gli inventari in [capability matrix](docs/capability-matrix-v1.md) sono evidenza disponibile. I [contratti runtime](docs/runtime-contracts-v1.md) vanno ristretti dopo la selezione, non implementati integralmente prima del confronto.

## B — Misurare e scegliere (P02–P03)

| Stato / ID | Azione | Dipendenze | Risultato e verifica |
|---|---|---|---|
| [ ] DW04.1 | Misurare il riferimento coding Q4 | DW03 | Esecuzione nel runtime di riferimento; qualità C#, startup, prefill, TTFT, decode, memoria e carico concorrente secondo protocollo. Separare cache fredda/calda e limiti CPU/GPU. |
| [ ] DW04.2 | Misurare APEX/MoE | DW04.1 | Confrontare configurazioni residenti e cache/caricamento selettivo disponibili. Registrare budget, I/O, hit/miss dove osservabili, contesa e qualità. Un percorso non disponibile è dichiarato tale; non simularlo e non costruire il runtime Satori per completare il confronto. |
| [ ] DW04.3 | Misurare Bonsai ternary | DW04.2 | Usare un runtime compatibile con PTQ1_0 e trasformazioni richieste. Misurare residency dove sostenibile, qualità e reattività. PQ2_0 richiede un seguito motivato da un limite specifico di PTQ1_0. |
| [ ] DW04 | Selezionare il percorso principale | DW04.3 | Tabella comparativa e verdetto sui criteri approvati. Selezionare un artefatto/famiglia e un riferimento esterno. Registrare capability indispensabili e politica di memoria. Se nessun candidato soddisfa i criteri, proporre riduzione del perimetro o arresto; non procedere tacitamente. |

Una configurazione oltre budget fallisce per quella configurazione. Il confronto di una configurazione alternativa deve essere identificato separatamente. Bassa VRAM non dimostra bassa memoria totale; pochi parametri attivi non dimostrano poco I/O.

## C — Implementare il percorso selezionato (P04–P05)

| Stato / ID | Azione | Dipendenze | Risultato e verifica |
|---|---|---|---|
| [ ] DW05 | Restringere contratti e task ai gap selezionati | DW04 | Capability matrix del percorso, contratti di sessione e risorse, piano memoria e sottotask ordinati DW06.1 ecc. Ogni sottotask ha dipendenze, oracle e criterio di verifica. Nessun obbligo di due architetture. |
| [ ] DW06 | Completare il percorso CPU | DW05 | Eseguire soltanto i sottotask approvati del percorso selezionato. Tokenizer/template conformi, codec/operatori contro reference indipendenti, intermedi/logits e generazione verificati. La checkbox si chiude solo a completamento dei sottotask. |

### Dipendenze condizionate alla selezione

| Percorso | Capacità da verificare o aggiungere |
|---|---|
| Coding Q4_K_M | Runtime denso e tokenizer necessari; Q4_K e Q6_K; tipi effettivi da inventario |
| APEX/MoE | Codec misti K/IQ effettivi; routing top-k ed esperti condivisi; backbone ibrido e stato; accesso alle slice degli esperti; cache/caricamento selettivo solo se richiesti dalla configurazione scelta |
| Bonsai PTQ1_0 | Codec Prism e BF16 richiesti; Hadamard, segni e semantiche embedding/output; backbone ibrido e stato; nessuna dipendenza da BitNet o da PQ2_0 |

AirLLM fornisce idee di lifecycle, caricamento per layer e prefetch. Il caricamento per layer non sostituisce la cache degli esperti. Implementare ciascuna politica soltanto quando la selezione ne dimostra la necessità. Non presumere identità degli operatori ibridi APEX/Bonsai.

## D — Risorse, ottimizzazione e consegna (P06–P08)

| Stato / ID | Azione | Dipendenze | Risultato e verifica |
|---|---|---|---|
| [ ] DW07 | Applicare i limiti e misurare Satori durante il lavoro | DW06 | Limiti di memoria, contesto e compute; cancellazione e recupero verificati. Workload concorrente contro baseline senza inferenza e runtime esterno. Pesi mapped non equivalgono a residency limitata. |
| [ ] DW08 | Completare cache/prefetch se selezionati | DW07 | Per percorso streamed: lease, pin, eviction in byte, backpressure, errori e cancellazione senza pesi parziali o cache hit falsi. Misurare byte/token, attese, overfetch e freddo/caldo. Per percorso residente: registrare non applicabile con motivo, senza aggiungere streaming. |
| [ ] DW09 | Ottimizzare il collo di bottiglia misurato | DW08 | Sottotask sequenziali per kernel/operatore con baseline e verifica end-to-end. SIMD e riuso delle attivazioni solo con parità numerica. Vulkan solo dopo criterio di beneficio fissato, con totale memoria iGPU e contesa inclusi. Nessun porting completo preventivo. |
| [ ] DW10 | Verificare la consegna | DW09 | Protocollo completo sul percorso scelto; requisito R-MM limitato alla famiglia selezionata; R-CAP, R-QUANT, R-PERF e R-RES con evidenze. Pubblicare artefatti supportati, comandi, limiti, qualità, risorse e motivi di rifiuto. Nessuna accettazione con soglie aperte. |

DW08 non applicabile non blocca DW09. DW09 può concludersi senza modifiche quando le misure non giustificano ottimizzazioni. L'assenza di GPU impedisce soltanto di dichiarare supporto Vulkan validato.

## Perimetro della consegna

- Un percorso principale selezionato e verificato, con confronto esterno riproducibile.
- Una taglia compatibile aggiuntiva solo per un uso osservato e autorizzato nel piano.
- Prove numeriche indipendenti e qualità coding separate.
- Budget, reattività, cancellazione e limiti operativi documentati.

Fuori perimetro: generalità multiarchitettura, BitNet obbligatorio, implementazione contemporanea APEX/Bonsai, codec non richiesti, UI, server, training, conversione ternary arbitraria, visione, KV su SSD, AVX-512, Metal e ROCm.

# Piano di implementazione per DeepSeek

## Mandato

Implementare Satori come motore .NET multi-modello per CPU Intel/AMD e Vulkan con budget iniziale di 16 GB RAM. Supportare pesi non-ternary e ternary compressi. Esplorare modelli oltre il working set residente mediante streaming dei layer e cache degli esperti MoE. Metal e ROCm sono memorandum, non dipendenze della consegna.

Requisiti vincolanti: [TASKS.md](TASKS.md), R-MM, R-CAP, R-QUANT, R-PERF. Questo piano è la sequenza operativa proposta per DeepSeek; non riscrive la cronologia T01–T45. Fonte delle idee: [applicabilità dei repository](docs/scouting/applicability-summary.md), [audit](docs/scouting/repository-findings-2026-10-13.md), [formati K/IQ](docs/scouting/quantization-candidates.md).

## Regole del worker

- Eseguire un task alla volta nell'ordine DS01–DS40. Ogni task dipende dal precedente; i riferimenti aggiuntivi identificano contratti da rispettare.
- Leggere istruzioni locali, stato Git e documentazione pertinente prima di modificare file. Il checkout contiene modifiche preesistenti e cancellazioni intenzionali: non ripristinarle, includerle nei commit o cancellarle senza attribuzione certa.
- Riutilizzare mapping, lettore GGUF, kernel scalar/AVX2 e primitive Vulkan esistenti. I precedenti runtime Vulkan nominali rimossi non sono capacità disponibili.
- Separare architettura, tokenizer, codec, trasformazioni, backend e politica di memoria. Vietare classi runtime chiamate Qwen, APEX o Bonsai quando esprimono contratti generici; identificatori di architettura e adapter specifici sono ammessi nel registro delle capability.
- Derivare forme, head dimension, epsilon, RoPE, tied weights, stato e workspace dai metadati validati. Non derivare sempre head_dim da hidden/heads.
- Mantenere i pesi compressi. Consentire scratch limitato per blocco/tile; vietare espansione permanente dell'intero modello in F32/FP16.
- Nessun PyTorch, Transformers, llama.cpp o servizio remoto come dipendenza del runtime produttivo. Sono ammessi strumenti esterni isolati per generare riferimenti indipendenti con provenienza e licenza.
- Non importare kernel senza verificare licenza e attribuzioni. AirLLM-plus resta fonte di idee finché la licenza non è chiarita.
- Test solo su invarianti critici: formati, numerica, SIMD/GPU, inferenza, memoria/lifetime, concorrenza e I/O. Per wiring, configurazione e documentazione usare build, demo o controlli mirati.
- Al completamento registrare esito, verifica e limiti in TASK-LOG.md; aggiornare la checkbox qui; creare un commit del solo task. Non fare push. Non dichiarare completato un task con verifica critica fallita o non eseguita.
- Usare `.temp/deepseek/<id>` per artefatti temporanei; non eliminare corpus e cartelle preesistenti. Tenere i pesi completi fuori dai commit. Scaricare solo gli artefatti necessari al task, dopo verifica di spazio, licenza e revisione.
- Se mancano hardware, reference o una decisione indispensabile, descrivere il blocco concreto. Chiedere soltanto la decisione mancante; non sostituire una misura con una stima. Nessun nuovo sistema ADR.

## Verifiche comuni

SDK da global.json: .NET 10.0.112, patch compatibili; C# 14. Build: `dotnet build Satori.sln -c Release --no-restore -m:1`; effettuare restore solo se necessario. Numerica: `dotnet test tests/Satori.Numerics.Tests -c Release --no-restore -m:1 --filter <test pertinenti>`. GPU: progetto `tests/Satori.Vulkan.Tests` con filtro pertinente e validation layer disponibile. Non copiare il placeholder del filtro: usare i test del task.

I comandi console/validation aggiunti devono avere help e una riga riproducibile nel log. Build senza nuovi warning. Un confronto deve usare lo stesso artefatto, tokenizer, contesto e sampling. Definire tolleranze numeriche per operatore, motivate dal riferimento, prima di ottimizzare. Non imporre uguaglianza dei token fra quantizzazioni diverse.

## A — Chiudere i contratti prima del runtime

DS02–DS03 sono task di specifica: DeepSeek opera qui come progettista e produce contratti correnti. L'implementazione dipendente parte solo quando R-CAP/R-MM sono coperti senza ambiguità.

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [x] DS01 | Esporre l'inventario GGUF usando il reader esistente | Comando end-to-end che mostra architettura, tensor type/shape/offset/byte e limiti su una fixture locale | Build e demo su fixture valida; test soltanto se si modifica parsing/range checking | Inferenza e download completo |
| [x] DS02 | Fissare la capability matrix dai metadati degli artefatti candidati | Matrice architettura/tokenizer/operatori/codec/trasformazioni/backend, con supportato/mancante e motivo; revisioni e provenienza | Inventari riproducibili di header/descrittori, nessuna capability dedotta dal nome file; eventuali campi ignoti espliciti | Kernel e promesse di qualità |
| [x] DS03 | Specificare runtime/sessione e piano memoria generici | Contratti per modello immutabile, sessione, stato KV/ricorrente, ownership, pesi condivisi, capability check, residency/prefetch e cancellazione | Due topologie rappresentabili senza dimensioni nominali; budget in byte senza doppio conteggio iGPU; invarianti di errore/lifetime scritti | Implementazione runtime |
| [x] DS04 | Fissare il protocollo di validazione | Corpus coding C#, italiano, output lunghi; riferimenti numerici; hardware, 2K/4K, misure e criteri | Manifest con versioni, comandi e provenienza; separazione correttezza/qualità/performance | Inventare soglie prodotto: chiedere valori minimi di qualità/velocità solo se necessari per un verdetto di accettazione |

Artefatti rappresentativi: Qwen2.5-Coder 1.5B e 7B per coding; Qwen3 4B per QK norm/head_dim esplicita; BitNet 2B per ternary W1.58A8; Tiel-Coder APEX per MoE ibrido/cache; Bonsai-2 27B per ternary g128/rotazioni. Le revisioni HF sono nelle evidenze dello scouting; risolvere la provenienza discordante APEX in DS02 prima di produrre reference. Non usare benchmark pubblicitari come oracle.

## B — Rendere generica l'inferenza CPU

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [x] DS05 | Implementare il descrittore validato di modello | Capability check e piano forme/memoria prima della generazione | Demo su due topologie; test critici per shape incompatibili, overflow e capability mancanti | Forward |
| [x] DS06 | Implementare il tokenizer BPE richiesto | Encoding/decoding, token speciali e template conformi agli artefatti scelti | Identità con sequenze ufficiali su italiano, codice, Unicode e token speciali | Tokenizer universale |
| [ ] DS07 | Implementare gli operatori CPU mancanti per Qwen2/Qwen3 | Bias pertinenti, GQA, QK norm, RoPE e proiezioni con dimensioni indipendenti | Reference indipendenti per ciascun operatore; riuso di quelli esistenti quando conformi | MoE e attenzione ricorrente |
| [ ] DS08 | Implementare il forward generico CPU | Logits mediante descrittore e registro operatori | Intermedi/logits conformi al riferimento su due topologie; Q4_0/Q8_0 già supportati | Sampling e streaming |
| [ ] DS09 | Implementare lo stato incrementale della sessione | KV dimensionata dal modello, reset, limiti e lifetime | Logits incrementali contro full recomputation; pieno contesto, reset, dispose e budget | Persistenza SSD della KV |
| [ ] DS10 | Collegare la generazione alla sessione | Prompt→testo, EOS, limite token, sampling esistente ove riutilizzabile | Sequenza greedy contro riferimento; seed verificato soltanto se sampling modificato | Server e UI |
| [ ] DS11 | Misurare la baseline coding residente | Report CPU su piccolo coding e 7B se il budget lo consente | Startup/prefill/TTFT/decode/picco memoria, 2K/4K, hardware e qualità; limiti espliciti | Ottimizzazioni prive di baseline |

## C — Ampliare i codec non-ternary

Ogni codec segue il layout upstream fissato in DS02; l'oracle deve provenire da codice/fixture indipendente, non dal decoder appena scritto. Il formato misto seleziona il kernel per tensore, mai per filename.

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [ ] DS12 | Implementare Q4_K scalar | Decoder e dot su blocchi compressi con scale/minimi | Blocchi noti, zeri, estremi e dimensioni valide; confronto FP64 sui pesi ricostruiti | SIMD |
| [ ] DS13 | Implementare Q6_K scalar | Decoder e dot registrati per tipo | Corpus indipendente e controlli di range | Nuove ricette |
| [ ] DS14 | Eseguire Q4_K_M end-to-end | Import di tutti i tipi dell'artefatto selezionato e generazione | Inventario completo senza fallback silenzioso; logits/reference e confronto qualità Q8 | IQ |
| [ ] DS15 | Ottimizzare Q4_K/Q6_K AVX2 | Dot specializzati con fallback scalar | Parità numerica; benchmark medesime matrici; mantenere experimental se nessun vantaggio | AVX-512 |
| [ ] DS16 | Implementare IQ3_XXS scalar | Ricostruzione codebook/segni e dot compressi | Oracle indipendente, indici e scale estremi; file non valido rifiutato | IQ2 |
| [ ] DS17 | Eseguire UD-IQ3_XXS selezionato | Tutti i tipi misti dell'artefatto inventariati e implementati con prove prima dell'uso | Stesso modello Qwen3 in Q8/Q4_K_M/IQ3; qualità e memoria separate dalla numerica | Dichiarare UD come codec unico |
| [ ] DS18 | Implementare Q3_K scalar | Supporto delle proiezioni richieste da APEX | Decoder/dot contro reference indipendente | Intero MoE |
| [ ] DS19 | Implementare IQ2_XXS scalar | Dot con codebook e scale corretti | Corpus indipendente e benchmark traffico/unpacking | Ternary |
| [ ] DS20 | Implementare IQ2_S scalar | Tipo distinto da IQ2_XXS, registrato nel dispatch | Reference di layout/segni/scale e rifiuto input invalidi | IQ1 e ricette non richieste |

Se DS02/DS17 individuano ulteriori tipi necessari, inserire prima dell'integrazione un task codec con decoder, dot e verifica indipendente; non accettare file parzialmente supportati. IQ1 resta fuori dal piano esecutivo.

## D — Validare il ternary nativo

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [ ] DS21 | Implementare I2_S richiesto da BitNet | Codec distinto da STQ2/STB1, con scale e layout fissati | Fixture indipendenti e dot numericamente verificato | Conversione arbitraria di modelli ordinari |
| [ ] DS22 | Implementare gli operatori W1.58A8 mancanti | Quantizzazione attivazioni, BitLinear, subln e ReLU² conformi al profilo | Intermedi contro riferimento ufficiale; tokenizer Llama3 verificato prima dell'integrazione | Assimilare BitNet a Transformer Qwen standard |
| [ ] DS23 | Eseguire BitNet end-to-end | Sessione generica ternary senza espansione completa | Logits/generazione contro oracle, picco RAM e qualità; R-QUANT insieme alla catena B | Bonsai |

## E — APEX: MoE ibrido con cache e streaming

APEX è una miscela di F32/Q8_0/Q3_K/Q4_K/Q6_K/IQ2_S/IQ2_XXS/IQ3_XXS secondo la scheda; DS02 deve verificarla nel binario. I 12,554 GB di pesi non impongono residency completa. Gli esperti attivi riducono il calcolo, non l'archiviazione totale. Il routing corrente precede la conoscenza degli esperti richiesti.

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [ ] DS24 | Implementare il routing MoE CPU | Top-k, pesi di routing, esperti condivisi e dispatch per slice | Oracle con router, tie e selezione esperti; forme reali derivate dal manifest | Cache SSD |
| [ ] DS25 | Implementare lo stato dell'attenzione ibrida APEX | Operatori DeltaNet/convoluzione/gating e full attention secondo architettura verificata | Reference multi-token, reset e confronto incrementale | Riutilizzare formule KV dense senza verifica |
| [ ] DS26 | Integrare il forward MoE ibrido CPU | Esecuzione con provider di pesi iniettato, stesso contratto sessione | Intermedi/logits contro riferimento; usare hardware adeguato o fixture ridotte per numerica | Promettere residency su 16 GB |
| [ ] DS27 | Implementare il provider streamed limitato | Lettura range compressi, ownership e finestra layer/esperti con budget in byte | I/O reale su fixture; short read, errore, cancellazione, dispose e nessun peso parziale visibile | Prefetch speculativo |
| [ ] DS28 | Implementare la cache degli esperti | Stati di caricamento, pin durante uso, eviction LRU in byte | Richieste concorrenti, singolo caricamento, eviction con lease, fallimento mai cache hit | Politiche predittive complesse |
| [ ] DS29 | Implementare il prefetch con backpressure | Code limitate, associazione richiesta/risultato e doppio buffer entro budget | I/O fallito propagato, coda piena senza perdita dei blocchi necessari, teardown deterministico | Predire come certe le scelte del prossimo token |
| [ ] DS30 | Integrare la residency parziale APEX | Pesi condivisi/stato residenti se nel budget; esperti caricati su miss | Inferenza reale CPU su 16 GB; stessi logits del provider residente, nessuna espansione completa | Accettare paging incontrollato come budget rispettato |
| [ ] DS31 | Misurare il costo dello streaming | Report layer streaming e cache esperti con compute reale | Byte letti/token, hit/miss, overfetch, attese, picco RAM, TTFT/decode, cache fredda/calda dichiarate | `time.sleep` o bandwidth teorica come risultato |

Se DS30 supera 16 GB anche con cache minima, registrare i componenti responsabili e ridurre soltanto parametri autorizzati come il contesto del protocollo; non eliminare operatori né cambiare modello tacitamente. Non serve scaricare/espandere FP16 completo per eseguire il modello quantizzato.

## F — Bonsai: ternary g128 con rotazione

La revisione verificata pubblica PTQ1_0 (5,947 GB) e PQ2_0 (7,206 GB), non il selettore TQ1_0 richiesto inizialmente. PTQ1_0 è 28 byte/128 pesi; PQ2_0 è 34 byte/128 pesi nel fork Prism fissato. Tipi privati 143/142: verificare identità/schema e non interpretarli come formati upstream. La trasformazione Hadamard non è opzionale.

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [ ] DS32 | Implementare PQ2_0 scalar | Packing g128 con scale FP16 e dot compressi | Reference Prism, ordine elementi e scale verificati | Alias con STQ2 |
| [ ] DS33 | Implementare PTQ1_0 scalar | Packing base3 g128 e dot compressi | Fixture indipendenti incluse mappe elementi; parità pesi con PQ2_0 equivalente | Rinominare in TQ1_0 g256 |
| [ ] DS34 | Implementare le trasformazioni Bonsai | Hadamard/segnature/normalizzazione da metadati, incluse semantiche embedding/output | Reference di trasformazione e proiezioni; rifiuto rotazione sconosciuta | Inferire i segni dal nome modello |
| [ ] DS35 | Integrare il backbone ibrido Bonsai | Adapter dell'architettura verificata riusando solo operatori conformi di DS25 | Intermedi e stato incrementale contro reference Bonsai; forme specifiche da metadati | Presumere identità con APEX |
| [ ] DS36 | Eseguire Bonsai end-to-end CPU | Generazione PTQ1_0/PQ2_0 con stesso runtime generico | Numerica contro oracle; qualità, picco RAM e packing comparati a 2K/4K | Visione/mmproj e 262K come promessa |

## G — SIMD e Vulkan misurati

| Stato / ID | Azione e contesto | Risultato osservabile | Verifica di completamento | Esclusioni |
|---|---|---|---|---|
| [ ] DS37 | Ottimizzare i codec aggiuntivi CPU | AVX2 per IQ/ternary selezionati dai profili operatori; riuso input quantizzato solo se numericamente conforme | Parità scalar, fallback e benchmark end-to-end; una specializzazione per commit se task suddiviso | Ottimizzare tutti i codec senza misure |
| [ ] DS38 | Implementare i kernel Vulkan richiesti dal manifest scelto | Pesi compressi e scratch limitato, capacità hardware esplicite; K prima, IQ/Prism in sotto-task sequenziali | Ogni kernel contro CPU, validation layer e misure separate trasferimento/kernel | Porting automatico CUDA; supporto dichiarato dall'esistenza di shader |
| [ ] DS39 | Integrare l'esecuzione Vulkan generica | Sessione/residency condivise, fallback CPU dichiarato e dispatch prefill/decode | Logits e lifetime verificati; totale CPU/GPU a parità di modello; nessun vantaggio GPU presunto | Planner nominale per modello |
| [ ] DS40 | Pubblicare la verifica dei requisiti | Matrice finale supportato/sperimentale/bloccato con report riproducibili | R-MM: due topologie; R-QUANT: ternary/non-ternary; R-CAP: rifiuti espliciti; R-PERF: dati e confronto con soglie approvate | Dichiarare performance accettata senza soglie o misure |

DS37–DS39 sono contenitori di ottimizzazione: prima di eseguirli scomporre per singolo codec/operatore/backend in task numerati DS37.1, DS38.1 ecc., ciascuno con baseline, corpus, criterio e commit. L'ordine dei sotto-task è determinato dai report DS11/DS31/DS36, non dall'intuizione. L'assenza di GPU blocca solo la dichiarazione di supporto Vulkan validato, non autorizza a simulare misure.

## Consegna finale

- Comandi riproducibili per inventario, inferenza e benchmark.
- Capability matrix corrente con revisioni/licenze e motivi di rifiuto.
- Report CPU/Vulkan, memoria e I/O, con hardware e limiti dichiarati.
- Prove numeriche indipendenti per codec/operatori critici; qualità coding distinta dalla parità del kernel.
- Stato dei 40 task con evidenza e commit; nessun task incompleto segnato come riuscito.

Fuori scope: UI, server, training, conversione ternary arbitraria, IQ1, vision tower, KV su SSD, AVX-512, Metal e ROCm. Questi punti richiedono un piano successivo.

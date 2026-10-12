# Piano di lavoro Satori

Obiettivo finale: generare testo da un modello GGUF supportato con un runtime .NET indipendente, pesi memory-mapped, backend CPU verificato e accelerazione Vulkan misurata. Target iniziale: mini PC Intel/AMD con 16 GB di RAM. Metal e ROCm restano estensioni future.

Il supporto fino a 27B è un obiettivo condizionato da modello, quantizzazione, contesto e memoria disponibile; non implica una velocità garantita. Il ternary sperimentale non implica che un modello convenzionale convertito mantenga la qualità originale.

## Regole di esecuzione

Stato: **T01–T37 completati**. GGUF Q8_0/Q4_0 su mapping e kernel scalar/AVX2 verificati anche su pesi pubblici reali; TernaryBitplane convertibile senza perdita e confrontato con Ternary2. Il bitplane non mostra un vantaggio generale su Ternary2 AVX2. T04 è ridotto al contratto concreto CPU descritto sotto. Profilo del primo modello: [Stories15M Q8_0](docs/stories15m-profile-v1.md), con fixture indipendenti di tokenizer, logits e stati intermedi. Operatori CPU del profilo implementati e verificati singolarmente: [contratto](docs/stories15m-operators.md). Tokenizer del profilo verificato sui sedici casi ufficiali. Forward sul prefisso verificato contro stati e logits indipendenti: [evidenze](docs/stories15m-forward.md). Cache incrementale con reset e limite 128 verificata contro il ricalcolo completo: [contratto](docs/stories15m-cache.md). Generazione greedy da console conforme alle sequenze ufficiali: [avvio e contratto](docs/stories15m-generation.md). Sampling con temperature, top-k/top-p e seed verificato. Baseline CPU/Vulkan e planner completati; [diagnostica T32](docs/vulkan-call-phases.md). Prossimo: **T38**. Evidenze in [TASK-LOG.md](TASK-LOG.md).

Politica di verifica aggiornata: niente test per bootstrap, configurazione o passaggi semplici; usare build e demo. Aggiungere test soltanto per invarianti critiche di formato, correttezza numerica, memoria/lifetime, SIMD, GPU e inferenza.

Eseguire un task alla volta, nell'ordine indicato. Ogni task dipende dal precedente, salvo T01. Segnare completato soltanto dopo la verifica. I task di decisione producono una specifica prima dell'implementazione dipendente. Nessun task include funzionalità delle fasi successive.

## A — Laboratorio numerico CPU

Contesto: .NET 10, C# 14, x64; xUnit per correttezza, BenchmarkDotNet per prestazioni. Nessuna dipendenza da altri runtime di inferenza. Vulkan, tokenizer e generazione sono esclusi da questa fase.

| ID | Task | Risultato atteso | Verifica | Esclusioni specifiche |
|---|---|---|---|---|
| T01 | Crea una soluzione eseguibile minima | Libreria Satori, console con prodotto matrice-vettore FP32 su dati noti, progetto test, SDK fissato | Build Release, test e console restituiscono il risultato atteso | Quantizzazione |
| T02 | Specifica il formato Ternary2 v1 | Specifica byte-level: ordine dei pesi, codici 00/01/10, rifiuto di 11, scale FP32, blocchi, padding, dimensioni, overflow e valori non finiti; regola deterministica di quantizzazione | Esempi binari calcolabili a mano coprono blocchi pieni, parziali e nulli | SIMD |
| T03 | Implementa il codec Ternary2 | Packing, unpacking e quantizzazione conformi a T02 | Test su tutti i codici, scale, padding, input invalidi e determinismo | MatVec |
| T04 | Definisci il contratto delle matrici CPU | Matrice Ternary2 con forma e block size immutabili, payload privato posseduto, importazione tramite copia validata; MatVec tramite span CPU | Build e utilizzo end-to-end nella console; validazione payload coperta dai test critici del codec | Descrittori generici e astrazione GPU rinviati all'introduzione di altri formati |
| T05 | Implementa la MatVec Ternary2 scalar | Calcolo diretto sui pesi compressi con accumulo FP32, senza espansione completa | Confronto con reference FP64 sui medesimi pesi ricostruiti; tolleranze documentate | AVX2 |
| T06 | Crea il corpus numerico riproducibile | Matrici con seed fisso, zeri, segni alterni, scale estreme ammesse e dimensioni irregolari | Test distinguono errore del kernel da errore di quantizzazione FP32→ternary | Qualità linguistica |
| T07 | Aggiungi il benchmark scalar | Report per dimensione: latenza, byte/peso incluse scale e padding, banda effettiva, allocazioni, ambiente | Esecuzione Release su corpus fisso; nessuna allocazione gestita nel percorso caldo dopo warm-up | Obiettivi arbitrari di velocità |
| T08 | Implementa la MatVec AVX2 | Kernel con code parziali, rilevamento capacità CPU e fallback scalar | Stesso corpus di T06; esecuzione con AVX2 e fallback; confronto prestazioni con T07 | AVX-512 |
| T09 | Implementa il proprietario memory-mapped | Mapping read-only con lifetime protetto durante l'esecuzione, accesso a file grandi e controlli degli intervalli | Test su file temporanei, offset, disposal, acquisizioni concorrenti e identità dei risultati | Prefetch specifico dell'OS |
| T10 | Collega MatVec ai pesi mappati | Console carica una matrice Ternary2 da file ed esegue scalar o AVX2 | Risultati identici alla versione in memoria; benchmark separa apertura/validazione, primo calcolo dopo validazione e regime caldo | GGUF completo; cold storage reale non misurato |

Traguardo A: una matrice FP32 viene quantizzata, salvata, mappata e moltiplicata con risultati verificati e prestazioni misurate.

## B — Formati reali

Contesto: il laboratorio A è funzionante. Importare non significa convertire automaticamente un formato GGUF in ternary. Il runtime Transformer rimane escluso.

| ID | Task | Risultato atteso | Verifica | Esclusioni specifiche |
|---|---|---|---|---|
| T11 | Specifica il sottoinsieme GGUF supportato | Versioni, tipi metadata, allineamenti, dimensioni tensoriali, limiti del parser e formati Q4_0/Q8_0 fissati | Fixture valide e casi da rifiutare documentati contro la specifica del formato | Supporto GGUF universale |
| T12 | Implementa il lettore GGUF | Lettura metadata e indice tensori su mapping, con validazione dimensioni e offset | Test su fixture note, file troncati, metadata invalidi e intervalli fuori file | Inferenza |
| T13 | Implementa la MatVec Q8_0 | Decodifica per blocco e kernel scalar su tensori importati | Confronto con valori di riferimento indipendenti e corpus reale | Q4_0 |
| T14 | Implementa la MatVec Q4_0 | Kernel scalar conforme al layout GGUF | Confronto indipendente e benchmark sullo stesso input di Q8_0 | Q4_K e Q2 |
| T15 | Ottimizza i kernel Q8_0/Q4_0 per AVX2 | Implementazioni specializzate registrate per formato | Test reference e benchmark ripetibili; mantenere sperimentali i kernel senza vantaggio dimostrato | Nuovi formati |
| T16 | Specifica TernaryBitplane v1 | Layout, scale, padding e conversione senza perdita da Ternary2 definiti | Esempi binari e identità dei pesi logici | Base3 |
| T17 | Implementa TernaryBitplane | Codec e MatVec scalar conformi a T16 | Confronto con gli stessi pesi di Ternary2 e benchmark completo della decodifica | LUT |
| T18 | Ottimizza TernaryBitplane per AVX2 | Kernel SIMD sul layout compatto | Corpus numerico e confronto equo con Ternary2 a scale identiche | Quantizzazione attivazioni |

Traguardo B: benchmark su pesi reali GGUF Q4_0/Q8_0 e confronto controllato fra layout ternari.

## C — Primo modello funzionante

Contesto: scegliere un solo modello piccolo e una sola architettura per il primo percorso completo. Nessuna interfaccia grafica; console sufficiente.

| ID | Task | Risultato atteso | Verifica | Esclusioni specifiche |
|---|---|---|---|---|
| T19 | Fissa il profilo del primo modello | Modello/versione/licenza, tokenizer, topologia, formati necessari, chat template e budget RAM documentati; fixture di token/logits con provenienza | Tutti gli operatori necessari sono elencati; formati mancanti diventano task obbligatori prima di T20 | Altre famiglie di modelli |
| T20 | Implementa gli operatori del profilo | Operatori CPU richiesti da T19, verificati singolarmente | Confronti con fixture indipendenti per normalizzazione, RoPE, attention e attivazioni effettivamente richieste | Operatori non richiesti |
| T21 | Implementa il tokenizer del profilo | Codifica, decodifica, token speciali e template del modello | Identità con sequenze ufficiali su corpus multilingue e casi limite | Tokenizer universale |
| T22 | Implementa il forward di un token | Logits ottenuti attraversando tutti i layer dai pesi GGUF | Confronto dei tensori intermedi e dei logits con fixture T19 | Generazione autoregressiva |
| T23 | Implementa la KV cache | Stato incrementale con limiti di contesto e budget memoria espliciti | Logits incrementali confrontati con ricalcolo completo; reset e contesto pieno verificati | Persistenza su SSD |
| T24 | Implementa la generazione greedy | Console prompt→token→testo con EOS e limite token | Sequenza deterministica confrontata con riferimento; errori leggibili per modelli non supportati | Sampling casuale |
| T25 | Aggiungi il sampling configurabile | Temperature, top-k/top-p e seed con regole di validazione | Test di distribuzione, riproducibilità e casi limite | API server |
| T26 | Pubblica il benchmark del primo modello | TTFT, prefill, decode, memoria e qualità rispetto a riferimento con stesso modello/configurazione | Report riproducibile con hardware, runtime, contesto e quantizzazione; nessuna misura stimata presentata come reale | Generalizzazione a 27B |

Traguardo C: Satori genera testo corretto da un modello piccolo supportato su CPU.

## D — Vulkan

Contesto: CPU verificata come riferimento; disponibilità di una GPU compatibile necessaria per chiudere la validazione hardware.

| ID | Task | Risultato atteso | Verifica | Esclusioni specifiche |
|---|---|---|---|---|
| T27 | Fissa il contratto del backend Vulkan | Binding, compilazione SPIR-V, ownership buffer, submission, sincronizzazione e capability minime documentati | Specifica comprende fallimento di inizializzazione, disposal e fallback CPU | Metal/ROCm |
| T28 | Implementa il percorso Vulkan minimo | Upload, dispatch di un kernel semplice e readback con lifetime espliciti | Risultato noto corretto su GPU; validation layer senza errori | Kernel quantizzati |
| T29 | Implementa la MatVec quantizzata Vulkan | Primo formato scelto fra quelli usati dal modello T19, con pesi residenti nei buffer | Corpus CPU/GPU entro tolleranza; tempi kernel separati dalla latenza end-to-end | Tutti i formati contemporaneamente |
| T30 | Integra l'offload del modello | Esecuzione GPU di operazioni o layer completi compatibili; fallback CPU | Logits conformi al riferimento; report dei trasferimenti, della memoria e della latenza totale | Divisione di una MatVec CPU/GPU |
| T31 | Implementa il planner misurato | Scelta di kernel/layout/backend basata su capacità e misure; cache invalidabile per hardware, driver e versione kernel | Scelte riproducibili, cache invalida respinta, confronto con CPU-only | Promessa di vantaggio GPU su ogni hardware |

Traguardo D: accelerazione Vulkan utilizzabile con vantaggi e limiti documentati sull'hardware provato.

## D2 — Ottimizzazioni Vulkan

T32–T37 completati; T38–T45 pianificati. [Startup T33](docs/stories15m-planner-startup.md). [Piano con dipendenze e verifiche](docs/vulkan-optimization-plan.md). Ogni task completato viene registrato in un commit separato.

## E — Estensioni dopo la baseline

Questi task sono in ordine indicativo. Prima di implementarli, fissare il profilo o il formato specifico e scomporre ciascuna estensione in task con fixture, kernel, integrazione e benchmark come nelle fasi precedenti.

1. Aggiungere i formati Q2/Q4_K richiesti dai modelli selezionati.
2. Valutare TernaryBase3/LUT su identiche matrici e scale.
3. Validare un modello nativamente ternario prima di estendere il runtime.
4. Valutare conversioni ternarie post-training con metriche di qualità indipendenti.
5. Aggiungere profili 7–9B e successivamente 20–27B entro budget misurati.
6. Valutare AVX-512 dove disponibile.
7. Specificare un backend Metal per Apple Silicon, incluso il percorso CPU ARM64.
8. Specificare un backend ROCm sulle GPU AMD effettivamente supportate.

Metal e ROCm sono memorandum, non dipendenze per completare la prima versione.

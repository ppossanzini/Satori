# Verification history

Historical task entries describe their original scope and may mention removed implementations. Current product requirements and executable work are defined only in TASKS.md.

# DeepSeek Plan Execution

Tasks DS01–DS40 from DeepSeek_todo.md. One task per commit; no push.

## DS01 — 2026-10-13 — Completed

Exposed the descriptor inventory of a GGUF container through `GgufInventory` (reusing `MappedWeights` and `GgufCursor`) and the console command `inventory <file.gguf> [--json]`. It reports architecture, version, alignment, data offset, tensor and metadata counts, every metadata key with its declared type and decoded scalar value or array shape, and every tensor with type, dimensions, declared offset, computed byte length and support status. `GgufTypes` adds the GGML type catalogue with block geometry and profile support.

Unsupported tensor types are reported explicitly instead of rejecting the file, which `GgufFile` still does for execution. The scan and the validated catalogue agree on the four golden fixtures (tensor name, type, dimensions, absolute offset, byte length, data offset, alignment, counts), checked in `GgufInventoryTests`.

Verification: Release build without warnings; 93/93 offline numerics tests passed (5 new); demo on `fixtures/gguf/v1/q8_0.gguf`, `f32.gguf` and the tokenizer fixture (arrays described, not materialised); non-GGUF input rejected with a readable message and exit code 1.

Reproduce: `dotnet run --project src/Satori.Console -c Release -- inventory fixtures/gguf/v1/q8_0.gguf`.

Limits: header, metadata and directory only; no weight decoding or execution; an unknown metadata value type is rejected because it cannot be skipped; private fork types (Prism PTQ1_0/PQ2_0) are not catalogued. Contract in `docs/gguf-inventory.md`.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.


## DS02 — 2026-10-13 — Completed

Pinned the first capability matrix from descriptor inventories of the candidate artifacts, without
downloading weights: a 32 MiB ranged download of each header/descriptor region, read by the DS01
inventory. Tool `tools/gguf_header_inventory.py`; per-artifact JSON and provenance in
`docs/scouting/inventories/`; matrix in `docs/capability-matrix-v1.md`.

Eleven artifacts at pinned revisions were inventoried: Qwen2.5-Coder 1.5B/7B (q4_0, q8_0, q4_k_m),
Qwen3 4B (Q4_K_M, Q8_0), BitNet b1.58 2B (i2_s), Tiel-Coder APEX-I-NanoPlus and Ternary-Bonsai-2 27B
(PTQ1_0, PQ2_0). Each inventory records architecture, tensor types with counts and `general.file_type`.

Findings: the official `q4_0` files carry a Q6_K `output.weight`, so capability is never deduced from the
file name. Unknown types are explicit: 36 (BitNet `I2_S`, 210 tensors), 143 (Prism `PTQ1_0`), 142 (Prism
`PQ2_0`); BF16 and the K/IQ types are listed as missing. The matrix covers architecture, tokenizer,
operators, codec and transformations with supported/missing and the reason, and lists four reader gaps
(`bitnet-b1.58` architecture string, `split.count = 0`, strict type set, private types).

Verification: inventory tool ran end-to-end; eleven inventories and the manifest produced; matrix written.

Reproduce: `python3 tools/gguf_header_inventory.py`.

Limits: descriptor region only; no weights, no decoding, no quality claim; APEX/Bonsai private types remain
unverified.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## DS03 — 2026-10-13 — Completed

Specified the generic runtime/session and memory-plan contracts in `docs/runtime-contracts-v1.md`: an
immutable `ModelDefinition` with descriptor-only content and metadata-derived dimensions; a capability
check producing explicit gaps before any weight read; a scoped `IWeightSource` with resident and streamed
implementations and shared/tied alias handling; `IInferenceSession` with KV and recurrent state layouts
selected per layer; a `MemoryBudget` with disjoint categories and `Unified`/`Discrete` domains that avoid
double counting on integrated GPUs; residency and prefetch; cancellation; and the error/lifetime
invariants.

Two topologies (dense grouped-query and hybrid MoE) are shown representable by the same definition with no
nominal dimensions; head dimension is taken from explicit `key_length`/`value_length` when present, never
from `embedding_length / head_count`. No runtime implementation was added; the requirement stays in
TASKS.md. Adapters are selected by `ArchitectureId` and never leak a model name into core interfaces.

Verification: two topologies represented without nominal dimensions; budget expressed in bytes with a
single owner per allocation and no iGPU double counting; error and lifetime invariants written.

Limits: specification only; DS05 onward implements it.

Skills: phase-development-project-conventions, base-dossier-writing.

# Registro di esecuzione

## T11 — 2026-10-12 — Completato

Skill: `phase-development-project-conventions`, `phase-development-task-execution`, `capabilities-dev-problem-solving-methodology`, `base-dossier-writing`. Scope: specifica GGUF, catalogo di accettazione e golden sintetiche. Nessuna modifica al runtime o nuovo test xUnit.

Contratto fissato in `docs/gguf-subset-v1.md`. Fonti ufficiali verificate alle revisioni riportate nel contratto. Il catalogo `docs/gguf-fixtures-v1.md` distingue casi validi, fuori profilo, corrotti, lifetime e prove numeriche future.

Quattro golden indipendenti dal codice Satori in `fixtures/gguf/v1/`, con generatore deterministico e manifest di hash, offset e risultati attesi. Verificati header, lunghezze, allineamenti, SHA-256, rigenerazione identica e decodifica indipendente dei payload. Q4_0/Q8_0 ricostruiscono gli stessi 32 pesi e producono −8 con un input di uno.

Nessun parser GGUF Satori o ufficiale è stato ancora eseguito sulle fixture. Le varianti del catalogo sono criteri per T12–T14, non test già passati. Nessuna build o suite di runtime ripetuta: il task modifica soltanto documenti e dati di riferimento. Controllo diff riuscito. Riferimenti temporanei rimossi dallo spazio di lavoro del task.

Prossimo: T12, nel confine di modulo e dipendenze fissato dal profilo.

## T10 — 2026-10-12 — Completato

Skill: `phase-development-project-conventions`, `phase-development-task-execution`, `capabilities-dev-problem-solving-methodology`. Quest'ultima usata per isolare un errore di sintassi nel test di file corrotti e un'eccezione non intercettata dalla console, correggendoli con verifiche mirate. Test limitati a formato binario, corruzione, identità numerica, disposal e allocazioni.

Implementato contenitore STQ2 v1, con header di 32 byte, forma, block size, versione e lunghezza payload. Salvataggio su file nuovo, senza sovrascrittura. Caricamento mapped con validazione completa una volta all'apertura. Forma immutabile interna condivisa; kernel scalar/AVX2 ora ricevono lo stesso span di payload, proveniente da array o mapping, senza duplicare algoritmi. Un callback con stato ref struct mantiene il mapping acquisito per l'intera MatVec senza closure o copia dei pesi.

Verifiche finali: build completa Release senza warning/errori; 24/24 test passati con AVX2 attivo e 24/24 con AVX2 disattivato. Cinque nuovi casi xUnit attraversano tutto il corpus con i tre block size, verificano header contro una fixture binaria indipendente, rifiuto di header/payload corrotti o troncati, identità bit per bit dei risultati mapped/owned per ciascun kernel, zero allocazioni dopo warm-up e rifiuto dopo Dispose. I test di T09 continuano a verificare il lifetime concorrente del proprietario riutilizzato.

Console: salvataggio del file demo e caricamento mapped riusciti; risultato `[4, -2.6666667]` con AVX2 e scalar. Rifiuto di un file non STQ2 con messaggio leggibile e codice di uscita 1, senza stack trace. Dopo la correzione del messaggio, build mirata della console nuovamente riuscita. Il primo tentativo di build dei nuovi test era fallito; i 19 test del vecchio assembly eseguiti allora non sono stati usati come evidenza di T10. Le prove finali sopra riguardano il nuovo assembly con 24 test.

Benchmark: 6/6 misure warm su dimensioni 1024²/2048²/4096² con blocchi 64, 9/9 prove di apertura con identità confermata e 72/72 verifiche numeriche della reference CPU. Zero allocazioni gestite per MatVec in entrambi i percorsi. Per 4096², circa 3,75 ms mapped e 3,76 ms owned in questa esecuzione: comparabili entro la variabilità, senza affermare vantaggi piccoli. Snapshot in `docs/benchmarks/mapped-2026-10-12/`.

Precisazione del criterio di benchmark: file appena scritti e page cache non svuotata; la validazione attraversa il payload. Sono distinti apertura/validazione, primo calcolo successivo e regime caldo; non è stata misurata una lettura SSD realmente fredda. Il criterio T10 è stato reso esplicito nel piano per evitare tale attribuzione.

Limiti: una matrice per file, payload massimo `int.MaxValue` byte, file immutabile richiesto; nessun checksum, salvataggio atomico o GGUF. Fixture e benchmark temporanei in sottodirectory uniche di `.temp/`, ripulite dai rispettivi proprietari. Il file demo è un output rigenerabile in `reports/`. Il formatter della console non poteva aprire il socket locale nel sandbox: indentazione corretta direttamente, ricompilazione riuscita.

Traguardo A completato. Prossimo T11: fissare il sottoinsieme GGUF prima di implementare il lettore.

## T09 — 2026-10-12 — Completato

Skill: `phase-development-project-conventions`, `phase-development-task-execution`. Scope: proprietario generico del mapping; test solo per invarianti critiche di puntatori, lifetime, concorrenza e allocazioni. Nessuna modifica ai kernel.

Implementato `MappedWeights`: apertura read-only, processo a 64 bit, offset `long`, finestre di lunghezza `int`, controlli prima dell'esposizione del puntatore. Accesso tramite callback con stato generico anche ref struct; nessuno span restituito. Acquisizione del SafeHandle sotto lock breve, callback fuori dal lock, rilascio del puntatore in `finally`. La chiusura respinge nuove letture, mentre quelle attive mantengono il mapping fino al proprio termine. Risorse parziali del costruttore rilasciate in caso di errore.

Verifica: build Release con zero warning/errori; 19/19 test critici complessivi passati. Sei nuovi casi coprono file sparse con lettura oltre 2 GiB, offset non allineati e intervalli invalidi, due lettori concorrenti attivi durante Dispose, Dispose interno alla callback, eccezione della callback e zero allocazioni dopo warm-up con delegate riutilizzato. Identità dei byte confrontata con contenuto noto del file.

Fixture create esclusivamente sotto `.temp/t09/<id>/` e rimosse dai rispettivi proprietari. Nessun file temporaneo dell'agente fuori dal progetto.

Limiti: verifica su Linux x64; file immutabile richiesto, nessuna garanzia di residenza RAM, nessun prefetch o buffer GPU. Il callback non trattiene span oltre il proprio termine. Integrazione numerica e benchmark del mapping rinviati a T10. Contratto in `docs/mapped-weights.md`.

## T08 — 2026-10-12 — Completato

Skill applicate: `phase-development-project-conventions`, `phase-development-task-execution`; test nuovi limitati alla correttezza SIMD, alle letture nelle code e alle allocazioni, come richiesto dall'utente.

Kernel AVX2: decodifica otto pesi da due byte tramite shift variabili sulle lane, maschere zero/segno sulle attivazioni, accumulo FP32 e riduzione per blocco. Nessun buffer di dequantizzazione. Code da uno a sette elementi elaborate scalar; load vettoriali solo entro il limite dell'input. Nessuna dipendenza FMA o AVX-512. Selezione automatica tramite capacità del runtime, fallback scalar; reference scalar ancora richiamabile esplicitamente.

Verifiche finali: build Release senza warning/errori; 13/13 test critici passati con AVX2 attivo e altri 13/13 con `DOTNET_EnableAVX2=0`. I test includono tutti i block size, 27 casi critici e dimensioni che attraversano code SIMD e limiti dei blocchi. Contatore del thread: zero allocazioni gestite dopo warm-up. Demo finale identifica correttamente AVX2 e scalar nei rispettivi processi, con stesso output noto.

Prestazioni: due esecuzioni separate, ciascuna con 18 benchmark e 72/72 risultati numerici validi. Vantaggio osservato 27,27–31,88× rispetto al kernel scalar di Satori su Intel Core Ultra 7 155H, per matrici 1024²/2048²/4096² e blocchi 64/128/256. Zero allocazioni per operazione in entrambi i kernel. Snapshot completi in `docs/benchmarks/avx2-2026-10-12/run1/` e `run2/`.

Limiti: job breve in-process, priorità ordinaria perché il sistema nega quella alta; risultati sintetici relativi a questa CPU. Nessun confronto con altri motori, nessuna misura token/s. La selezione automatica usa la disponibilità AVX2; un futuro planner potrà scegliere per forma/hardware. Prossimo task T09: ownership e lifetime del mapping.

## T07 — 2026-10-12 — Completato

Skill: `phase-development-project-conventions`, `phase-development-task-execution`; nessun nuovo test sui passaggi semplici di configurazione del benchmark.

BenchmarkDotNet 0.14.0 con toolchain in-process, un thread, tre warm-up e tre iterazioni misurate da circa 200 ms. Matrici 1024², 2048², 4096²; blocchi 64/128/256. Setup e quantizzazione esclusi; decodifica inclusa. Report JSON con latenza, deviazione standard, MatVec/s, byte/peso, banda effettiva e allocazioni. Report numerico esteso anche alle nove matrici misurate.

Verifica: build Release riuscita senza warning/errori; 36/36 combinazioni numeriche passate; 9/9 benchmark conclusi; zero allocazioni gestite per operazione riportate dal diagnoser. Snapshot in `docs/benchmarks/scalar-2026-10-12/`; metodo in `docs/benchmark-method.md`. Report rigenerabili sotto `reports/scalar/`, esclusi da Git.

Limite osservato: priorità alta negata dal sistema e notevole variabilità, con intervalli di confidenza ampi nel job breve. Baseline preliminare; nessuna scelta di block size o promessa di prestazioni. Il confronto con AVX2 richiederà misure ripetute con lo stesso protocollo. Nessun test di modello o GPU.

## T06 — 2026-10-12 — Completato

Skill: `phase-development-project-conventions`, `phase-development-task-execution`; prosecuzione autorizzata dall'utente, test limitati agli invarianti numerici critici.

Corpus v1 con xorshift32, seed 1729 e conversione FP32 fissa; nove famiglie per ciascun block size: casuale, zeri, alternanza, sparse, scale piccole/grandi, subnormal, scala massima e singolo peso. Righe e blocchi parziali inclusi. Reference FP64 separata per pesi originali e ricostruiti; report distingue errore sui pesi, errore di quantizzazione sull'output ed errore del kernel. Tolleranza derivata dal bound di accumulo FP32 con termine per underflow.

Verifica: build Release senza warning/errori; 9 casi xUnit complessivi passati, inclusi tre casi che attraversano le 27 combinazioni del corpus. Nessuna modifica al kernel. Il corpus non misura qualità linguistica o velocità.

## Revisione verifiche — 2026-10-12

Su richiesta dell'utente, eliminati i tre test ovvi FP32 e il progetto Satori.Tests. Il risultato storico T01 sotto descrive la verifica effettuata allora, non test ancora presenti. Le verifiche future dei passaggi semplici usano build/demo; i test sono riservati ai passaggi critici.

## T02–T05 — 2026-10-12 — Implementati

Skill applicate: `phase-development-project-conventions` e `phase-development-task-execution`. L'istruzione esplicita dell'utente di procedere con implementazioni e test solo critici governa questo incremento, senza ulteriori gate di approvazione.

- T02: specifica Ternary2 v1 fissata con esempio binario indipendente; scale per blocco, padding, soglia, valori non finiti e limiti espliciti.
- T03: codec con quantizzazione, importazione validata e dequantizzazione; test binari e numerici critici superati.
- T04: contratto concreto CPU con payload privato posseduto e copia all'importazione. Semplificazione del piano: descrittori generici rinviati finché esiste un solo formato quantizzato; nessun puntatore o lifetime esterno introdotto.
- T05: MatVec scalar sui blocchi compressi, senza espansione dei pesi. Confronto FP64 indipendente sui medesimi pesi quantizzati, su righe e blocchi parziali per tutti i block size. L'errore rispetto alla matrice FP32 originale rimane una misura distinta da sviluppare nel corpus T06.

Verifiche: build Release senza warning/errori; 6 casi critici xUnit passati; zero allocazioni gestite misurate su 100 esecuzioni dopo warm-up; console con output `[4, -2.6666667]`, 40 byte incluse scale/padding; controllo diff senza errori. Restore e test runner eseguiti con accesso autorizzato per rete/socket locale.

File principali: `docs/ternary2-v1.md`, `src/Satori.Core/Ternary2Matrix.cs`, `src/Satori.Core/Ternary2MatVec.cs`, `src/Satori.Console/Program.cs`, `tests/Satori.Numerics.Tests/`.

Limiti: memoria gestita posseduta, non ancora mapping; nessun benchmark di velocità, SIMD o qualità di modello. T06–T07 restano aperti.

## T01 — 2026-10-12 — Completato

Scope: soluzione, libreria FP32, console, test, configurazione SDK e istruzioni di avvio.

Skill dichiarata: `phase-development-project-conventions`, per adottare struttura e verifiche del progetto. Nessuna skill di server/API: questo incremento è una libreria numerica locale.

Risultato: soluzione .NET 10/C# 14 con prodotto matrice-vettore FP32 row-major, validazione delle dimensioni e divieto di output sovrapposto agli input. Console con caso noto 2×3. Nessuna quantizzazione implementata in questo task.

Verifiche su Linux x64, SDK 10.0.112:

- `dotnet restore Satori.sln -p:RestoreDisableParallel=true`: riuscito con accesso di rete per NuGet.
- `dotnet build Satori.sln -c Release --no-restore -m:1`: riuscito, zero warning, zero errori.
- `dotnet test Satori.sln -c Release --no-build --no-restore -m:1`: 3 test passati; richiesto accesso al socket locale del test runner.
- `dotnet run --project src/Satori.Console -c Release --no-restore -p:BuildInParallel=false`: output `Satori FP32 MatVec: [9, 4]`, uscita 0.

I limiti del sandbox hanno impedito il primo restore e la prima esecuzione dei test; i tentativi autorizzati sono riusciti senza ridurre i controlli del progetto.

## T12 — Lettore GGUF

Implementato `Satori.Formats.Gguf`: cataloghi immutabili, scalar typed, array indicizzati, finestre tensoriali e mapping protetto. Validati UTF-8, tipi, dimensioni, overflow, allineamenti, intervalli e limiti del profilo. Core non dipende da GGUF.

Verifica: build Release senza warning; 46 test critici passati, inclusi golden, tutti i troncamenti della fixture Q8_0, metadati scalar/array, allineamenti 8/64/4096, file sparse oltre 2 GiB, overlap e lifetime. Corretto un offset errato nella fixture a 8 byte prima della chiusura. Il confronto con un reader ufficiale resta da integrare prima della consegna del gruppo T12–T18.

Skill: phase-development-project-conventions e phase-development-task-execution. Autorizzazione utente: avanzamento fino a T18 e test soltanto critici, senza gate intermedi. Non si applicano convenzioni API/server a questa libreria numerica.

## T13 — MatVec Q8_0 scalar

Core riceve payload e forma senza dipendere da GGUF; l'adapter valida le scale una volta, poi legge il tensore con lifetime protetto. Scala FP16 con segno, accumulo FP32 per blocco. Aggiunto validatore separato per reference FP64 indipendenti.

Verifica: build Release pulita, 51 test passati; golden −8, Int8 completo, scale negative/±0/subnormal, rifiuto NaN/Inf, owned/mapped e zero allocazioni. Provato il GGUF pubblico stories15M-q8_0: catalogo di 57 tensori confrontato col reader ufficiale fissato e MatVec di tre tensori reali confrontate con decodifica NumPy FP64. Report in docs/benchmarks/gguf-2026-10-12; nessuna inferenza eseguita. Le quattro golden sono state lette anche dal reader ufficiale, chiudendo la verifica aggiuntiva di T12.

Skill: phase-development-project-conventions, phase-development-task-execution. Script reference esterno soltanto nella validazione; nessuna dipendenza runtime da GGML.

## T14 — MatVec Q4_0 scalar

Aggiunto Q4_0 al contratto numerico e all'adapter: nibble bassi e alti nelle due metà distinte di 16 pesi. Stesse forme e attivazioni di Q8_0 nei confronti; nessuna conversione ternaria.

Verifica: build Release pulita, 55 test critici passati. Golden, tutti i nibble, scale con segno/subnormal e non finite. Catalogo ufficiale e tre MatVec FP64 sul GGUF stories15M-q4_0 verificati; zero allocazioni. Eseguiti sei benchmark BenchmarkDotNet scalar sui due file reali; report completi conservati nella cartella gguf-2026-10-12/scalar. Le latenze rapide del validatore sono diagnostiche; il riferimento prestazionale è BenchmarkDotNet.

Skill: phase-development-project-conventions, phase-development-task-execution.

## T15 — AVX2 Q8_0/Q4_0

Implementati kernel AVX2 separati dal contenitore, con conversione Int8/nibble in otto lane FP32, riduzione per blocco, senza FMA né espansione completa. Selezione automatica per formato quando AVX2 è disponibile; scalar come fallback.

Verifica: build Release pulita; 59 test passati con AVX2 e 59 con AVX2 disabilitato. Reference FP64 indipendenti su entrambi i file pubblici rispettate. Due esecuzioni BenchmarkDotNet, dodici misure ciascuna, zero allocazioni e vantaggio su tutte le forme misurate; confronto e ambiente in docs/benchmarks/gguf-2026-10-12. Il vantaggio autorizza il dispatch automatico su questo percorso, senza generalizzazioni a inferenza o hardware diverso.

Skill: phase-development-project-conventions, phase-development-task-execution; base-dossier-writing per specifiche e report concisi.

## T16 — Specifica TernaryBitplane v1

Fissati piani P/N, scala FP32, ordine LSB, bit sovrapposti invalidi, padding, zero canonico, contenitore STB1 e conversione bit-identica da/verso Ternary2. Stessi byte/peso: il confronto isola il layout. Esempio indipendente verificato: P=0x89, N=0x14, Ternary2=0x61/0x42, somma logica 1. Specifica in docs/ternary-bitplane-v1.md.

Verifica: esempio ricalcolato dalle otto posizioni, lunghezze dei blocchi 20/36/68 byte uguali al Ternary2. Nessun codice o test aggiunto in questo task di specifica.

Skill: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing.

## T17 — Codec e MatVec TernaryBitplane scalar

Implementati import validato con copia, conversione lossless da/verso Ternary2, salvataggio/load STB1 e MatVec scalar. Scale e codici immutati, senza seconda quantizzazione; nessuna espansione nel kernel.

Verifica: build Release senza warning, 64 test passati. Golden indipendente, bit sovrapposti/padding/scale invalidi, round trip byte-identico e risultati scalar identici su tutte le 27 combinazioni del corpus, zero allocazioni, ogni troncamento del contenitore golden rifiutato. Diciotto benchmark completi scalar per entrambi i layout, tre dimensioni e tre block size; snapshot in docs/benchmarks/bitplane-2026-10-12/scalar.

Skill: phase-development-project-conventions, phase-development-task-execution; base-dossier-writing per la specifica.

## T18 — TernaryBitplane AVX2

Implementato kernel a otto lane con estrazione congiunta P/N, segno dalle maschere e coda scalar entro i confini dell'input. Dispatch AVX2 con fallback; nessuna dipendenza FMA/AVX-512. Comparatore di layout verifica gli stessi pesi e scale prima del benchmark.

Verifica finale: build Release senza warning/errori; 68 test critici passati con AVX2 e 68 con AVX2 disabilitato. Corpus esteso alle code 1–17 e ai confini dei blocchi; identità SIMD con Ternary2 e confronto FP64. Validatore dei layout: 36/36 combinazioni passate, incluse nove matrici grandi. Due run BDN da 36 misure ciascuno, tutte senza allocazioni. Bitplane AVX2 20,2–39,2× sul proprio scalar; rispetto a Ternary2 AVX2 rapporto di latenza 0,64–1,54, migliore in 7/18 coppie: nessun vantaggio generale dimostrato. Ternary2 rimane il layout già adottato, bitplane disponibile come alternativa sperimentale. Report, ambiente e confronto in docs/benchmarks/bitplane-2026-10-12.

Revisione finale del gruppo: rafforzato il tipo di eccezione atteso nei test di corruzione GGUF; limite 1 TiB controllato prima del mapping e verificato su file sparse; validatore GGUF controlla SHA-256 e numero esatto delle voci reference. Cataloghi e MatVec dei due file reali già verificati. Aggiornati README, specifiche, istruzioni di riproduzione e stato del piano. Rimossi soltanto i temporanei di questo gruppo; corpus pubblico riscaricabile con hash fissati.

Skill: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. Autorizzazione utente: completamento sequenziale fino a T18 con test critici senza approvazioni intermedie. Nessun commit richiesto o creato. Prossimo task: T19, profilo del primo modello; nessuna inferenza implementata in questo gruppo.

## T19 — Profilo del primo modello

Fissato Stories15M Q8_0 con hash e revisione, dichiarazione MIT dell'autore originale, tokenizer SPM, topologia, assenza di chat template, contesto 128 e budget RAM calcolato. Q8_0 e F32 coprono tutti i 57 tensori; T20 include lettura F32 e gather Q8_0, senza nuovi formati quantizzati. Contratto in docs/stories15m-profile-v1.md.

Verifica: generatore C++ compilato contro llama.cpp fissato; 147 file fixture verificati e rigenerati identici byte per byte. Sedici casi tokenizer, tre prompt con logits completi e 15 stati intermedi ciascuno, tre sequenze greedy di 12 token. FP64 indipendente contro runtime ufficiale sugli stessi pesi ricostruiti F32: massimo errore logits <1e-5, intermedi <2,2e-5. Sequenze greedy identiche anche al percorso ufficiale Q8_0; scostamenti numerici di quest'ultimo conservati separatamente. Manifest con hash, provenienza, sorgenti e ambiente in fixtures/models/stories15m-q8-v1.

Nessun codice di inferenza Satori aggiunto. Nessun test ovvio o nuova dipendenza .NET. La verifica riguarda le fixture; T20–T24 devono ancora implementare e verificare operatori, tokenizer, forward, cache e generazione. Rimosso soltanto .temp/t19; corpus GGUF condiviso conservato per i prossimi task.

Skill: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. Prossimo: T20.

## T20 — Operatori CPU del profilo

Implementati RMSNorm con accumulo FP64, RoPE a coppie adiacenti/base 10000, attention sul prefisso causale con softmax stabile, SwiGLU stabile e somma residuale. MatVec Q8_0 riutilizzata. Gather di una sola riga Q8_0 dal mapping; norme F32 copiate e validate una volta. Buffer e scratch forniti dal chiamante, contratti di alias espliciti. Nessun tokenizer, forward o proprietario della KV cache introdotto.

Verifica: build Release senza warning/errori; 75 test critici passati (sette nuovi). Confronto per componente con fixture indipendente NumPy FP64 per normalizzazione, RoPE, attention e attivazione, tolleranza 2e-6 + 2e-6 * abs(reference). Casi estremi FP32, in-place, dimensioni/alias, Q8 signed e scala negativa, F32 non finiti e lifetime. Zero allocazioni misurate dopo warm-up per operatori e gather mappato. Fixture rigenerata identica e hash verificati. Una fixture GGUF inizialmente priva del metadata di quantizzazione è stata corretta; suite finale completamente verde. Nessun benchmark prestazionale aggiunto; questi operatori sono la baseline CPU numerica.

Contratto in docs/stories15m-operators.md; generatore e manifest versionati in tools/transformer_reference.py e fixtures/operators. Skill applicate: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. Esecuzione e verifiche critiche autorizzate dalla richiesta T20 e dalla politica test dell'utente. Prossimo: T21. Nessun commit richiesto o creato.

## T21 — Tokenizer del profilo

Implementato StoriesTokenizer con vocabolario posseduto, merge per score FP32 e posizione sinistra, byte fallback, BOS opzionale e nessun EOS automatico. Codifica UTF-16 rigorosa con limite 16384 byte UTF-8. Decodifica completa e continuazione UTF-8 con gestione dei byte incompleti. Nessun chat template nel profilo. Hash degli array, ID, tipi e opzioni vincolati al vocabolario Stories15M.

Verifica: build Release senza warning/errori; 79 test critici passati, quattro nuovi test. Identità di token, byte dei pezzi e decodifica sui sedici casi ufficiali. Verificati streaming multibyte, whitespace della continuazione, input massimo, surrogate isolate, ID invalidi, byte incompleti e vocabolario modificato. Tokenizer utilizzabile dopo disposal del mapping. Allocazioni totali di costruzione dopo warm-up entro 16 MiB. Fixture tokenizer offline di 723488 byte rigenerata identica, con hash e provenienza separati dagli oracoli completi.

Perimetro: StoriesTokenizer e StoriesTokenDecoder nel progetto GGUF, test critici, estrattore Python standard library, fixture/manifest e docs/stories15m-tokenizer.md. Nessuna nuova dipendenza runtime. Rimossa soltanto .temp/t21; corpus condiviso conservato. Nessun forward o generatore introdotto. Prossimo: T22.

Skill: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. La richiesta di procedere e la politica utente sui test critici autorizzano implementazione e verifica senza gate intermedi. Nessun commit richiesto o creato.

## T22 — Forward di un token

Implementato StoriesModel: caricamento dell'artefatto Q8_0 fissato con verifica SHA-256, metadata e 57 tensori; embedding, sei layer completi, norma finale e proiezione dei 32000 logits. Pesi mappati compressi e norme F32 possedute. Forward dell'ultimo token tramite ricalcolo del prefisso, senza stato persistente. K/V temporanei; cache incrementale e generazione escluse. Observer diagnostico con 15 stati transitori. Chiamate e disposal serializzati.

Verifica: build Release senza warning/errori; 88 test critici passati con AVX2 e 88 con AVX2 disabilitato. Nove nuovi casi di integrazione. Tutti i 15 intermedi e i 32000 logits dei tre prompt confrontati con NumPy FP64, sia scalar sia AVX2. Massimo errore assoluto 1,430831e-5; massimo RMS per tensore 2,654648e-6; argmax conformi. Verificati input invalidi prima della scrittura, assenza di stato residuo, recupero dopo eccezione observer, disposal e rifiuto di artefatti diversi. A 128 token: logits finiti e 1782640 byte temporanei misurati dopo warm-up, senza espansione completa dei pesi. Nessuna misura di throughput o working set dichiarata.

Perimetro: StoriesModel, StoriesLayer e StoriesStateObserver nel progetto GGUF; StoriesForwardTests e docs/stories15m-forward.md. Test di integrazione separati tramite categoria, corpus reale necessario e mai scaricato dai test. Le fixture esistenti restano invariate. Nessun temporaneo persistente aggiunto. Prossimo: T23.

Skill applicate: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing. La richiesta di procedere e la politica utente sui test critici autorizzano implementazione e verifica senza gate intermedi. Nessun commit richiesto o creato.

## T23 — KV cache incrementale

Implementata StoriesSession, creata dal modello e proprietaria di K/V FP32 e scratch riutilizzabili. Una sequenza per sessione, capacità fissa 128, K dopo RoPE e V senza rotazione. TryAppend elabora soltanto il nuovo token e avanza Count dopo il successo; a contesto pieno restituisce false senza cambiare logits o cache. Reset azzera e riusa i buffer. Dispose azzera e rilascia il workspace senza chiudere il modello condiviso. Forward completo conservato come riferimento separato.

Verifica: build Release senza warning/errori; 97 test critici passati con AVX2 e 97 con AVX2 disabilitato. Nove nuovi casi di integrazione. Logits incrementali esattamente uguali al ricalcolo per ogni token dei tre prompt, scalar e AVX2. Confronto lungo alle posizioni 0, 1, 31, 63 e 127; verificati limite pieno, reset verso prefisso diverso, errori senza avanzamento, lifetime e sessioni concorrenti indipendenti. Payload K/V 1769472 byte; costruzione sessione 1782808 byte misurati. Append e reset: zero allocazioni dopo warm-up, verificati anche nel test isolato.

Perimetro: StoriesSession, StoriesWorkspace, percorso incrementale in StoriesModel, test critici e docs/stories15m-cache.md. Nessun nuovo formato o dipendenza. Generazione e sampling esclusi. Nessun temporaneo persistente aggiunto. Prossimo: T24.

Skill: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; capabilities-dev-problem-solving-methodology per isolare la misura del primo reset e completare il warm-up del test senza cambiare la soglia. La richiesta di procedere e la politica utente sui test critici autorizzano implementazione e verifica senza gate intermedi. Nessun commit richiesto o creato.

## T24 — Generazione greedy

Implementati StoriesGenerator, opzioni centralizzate, risultato immutabile e motivi Eos/TokenLimit/ContextLimit. Prompt codificato con BOS, prefill e cache nuova per chiamata, argmax con parità all'ID minore e streaming della sola continuazione UTF-8. Default 32 nuovi token, budget limitato dallo spazio residuo entro 128. Console generate collegata al runtime GGUF; errori di modello, prompt e limite leggibili, con uscita 1.

Verifica finale: build Release senza warning/errori; 109 test critici passati con AVX2 e 109 con AVX2 disabilitato, PGO predefinito. Ulteriore run standard: 109 passati. Dodici nuovi casi: nove di generazione, due di argmax e una regressione parser. Tutti i 12 token dei tre oracoli ufficiali identici sia scalar sia AVX2; testo e streaming conformi. Verificati default, ripetibilità, budget zero, contesto pieno/quasi pieno, prompt eccessivo, callback fallito e lifetime. Le fixture non generano EOS nei 12 token: la selezione e la soppressione del pezzo speciale sono verificate separatamente; nessun arresto EOS end-to-end attestato su questi prompt.

Console verificata su completamento reale e messaggi negativi, incluso il ritorno alla demo Ternary2. Output di Once upon a time con 12 nuovi token: , there was a little girl named Lily. She loved. Conteggi: prompt 5, generati 12, TokenLimit. Nessun benchmark prestazionale o giudizio generale di qualità aggiunto.

Corretto un crash JIT emerso nella suite: GgufCursor.String usa un contenitore dedicato al posto dell'array temporaneo covariante. Diagnosi con dump e simboli del runtime installato, controllo PGO discriminante e regressione su 1000 letture. Workaround circoscritto; SDK e ottimizzazioni predefinite invariati. Evidenza in docs/diagnostics/dotnet-string-callback.md. Rimossa soltanto .temp/t24, inclusi dump, simboli e strumenti temporanei; corpus condiviso conservato.

Perimetro: selettore Core, generazione e contenitore stringhe GGUF, console con riferimento GGUF, test critici, docs/stories15m-generation.md e stato del piano. Nessuna nuova dipendenza runtime. Skill: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options per le opzioni di generazione, base-dossier-writing; capabilities-dev-problem-solving-methodology per il crash durante la verifica. La richiesta di procedere e la politica utente sui test critici autorizzano implementazione e verifica senza gate intermedi. Nessun commit richiesto o creato. Prossimo: T25.

## T25 — Sampling configurabile

Aggiunti SamplingOptions e TokenSampler nel Core; integrazione in StoriesGenerator e flag console temperature, top-k, top-p e seed. Default greedy preservato, opzioni validate prima del prefill anche a budget zero. Softmax FP64 stabile, top-k prima del nucleus, parità per ID e PRNG SplitMix64 fissato con seed unsigned a 64 bit ripartito a ogni generazione. Scratch posseduto dal sampler e ordinamento O(V log V); nessuna dipendenza aggiuntiva. Contratto e limiti di riproducibilità fra piattaforme in docs/stories15m-generation.md.

Verifica: build Release senza warning/errori; 114 test passati con AVX2 e 114 con AVX2 disabilitato. Cinque nuovi test critici: vettori PRNG noti, distribuzioni analitiche, filtri e limiti numerici, configurazioni/logits invalidi e integrazione riproducibile con streaming/scalar. Console reale ripetuta con seed 123 identica; sei casi negativi rifiutati. Oracoli greedy precedenti invariati. Nessun benchmark o nuova attestazione EOS end-to-end. Nessun temporaneo persistente creato; corpus condiviso conservato. Prossimo: T26.

Skill applicate: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing. La richiesta T25 e la politica utente autorizzano implementazione e sole verifiche critiche senza gate intermedi. Nessun commit richiesto o creato.

## T32 — GPU call phase diagnostics (completed)

Added opt-in host intervals for upload (map/copy/cache/unmap), command recording, submission preparation, QueueSubmit, fence wait, post-wait validation, query retrieval and readback. Successful MatVec counters expose submissions, waits, transferred bytes and instrumented-call counts. Explicit model options independently enable GPU timestamps and host phases; diagnostic options do not enter Auto planner selection.

Published six sequential process reports: three per timestamp mode, each with three warmups and seven measured runs on the existing eleven-token workload. Every checked prefix passes CPU/GPU numerical gates. Each measured run has 473 submissions/waits, 2,865,280 vector bytes, zero managed hot allocations and phase deltas summing to host duration within 1e-6 ms. Timestamp-off decode medians span 149.783–157.967 ms versus CPU 3.720–5.646 ms. Fence wait dominates; it overlaps device execution and cannot be equated to pure host overhead or transfer time. No speedup claimed. Report.

Release build: zero warnings/errors. Thirteen existing GPU/policy tests pass with Khronos SyncVal; 116 CPU tests pass separately. No new implementation-mirroring tests. Local validation package and scripts were confined to `.temp/t32` and removed after publication. Native shader/algorithm unchanged.

Skills: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing; capabilities-dev-problem-solving-methodology for diagnostic scope and evidence. User authorization covers critical verification alongside implementation and requests a commit per completed task. Prerequisite code baseline recorded separately as cc5cf8c, excluding pre-existing editorial/personal changes. T33 is next; T34–T45 remain planned.

## T33 — Reuse CPU plans before Vulkan weight uploads (completed)

Separated device/binary identity, cache lookup and calibration. Auto now returns a validated CPU model before constructing GPU matrices when the matching cache selects CPU. Cached GPU choices retain resident construction without recalibration; invalid/missing caches retain the numerical gate and measured selection. Required and unavailable-device behavior remain intact. Initialization snapshots expose candidate device and Q8 resource/work counters without requiring an active GPU model.

Published startup evidence: three independent measurement groups, each three discarded fresh-process warmups plus seven measured processes per version, alternating order. Baseline backend a704883, identical runner scope and separate binary-valid caches. Runtime initialization medians change 432.616→214.766, 418.548→217.648 and 414.409→234.402 ms (43.4–50.4% reduction). Every optimized CPU cache hit has zero matrices, weight/matrix allocation bytes and dispatches/transfers. Context creation remains; no inference speedup is claimed.

Real process checks cover fresh/identical/invalid identity/invalid samples/malformed cache, and an explicitly synthetic cached GPU choice (not performance evidence). Output is Once/token 9038 throughout successful runs. CLI checks Required, unavailable-driver Auto/Required and invalid model failure with a valid cache. Build: zero warnings/errors; two policy tests pass. CPU suite: 116 pass when run alone; first run had the previously observed transient allocation assertion (7,752 bytes), with 115 others passing. No numerical/native synchronization changes; T32 SyncVal evidence retained. No obvious new unit tests. Task-local baseline/caches/scripts removed after publication; shared corpus preserved.

Skills retained: phase-development-project-conventions, phase-development-task-execution, phase-development-configuration-options, base-dossier-writing and capabilities-dev-problem-solving-methodology. Standing authorization covers critical verification and a commit per completed task. Pre-existing editorial/personal changes excluded. T34 is next: define the resident GPU session contract before implementation.

## T34 — Define the resident GPU session contract (completed)

Pinned ownership, isolated KV/scratch, one submission per append, immutable recording descriptors, buffer budget, workgroup reduction, FP32 numerical gates and failure/reset/disposal transitions. Reviewed all operator dependencies and the six-layer token diagram against the CPU implementation. Documentation-only task; no runtime changes or tests. Contract: docs/adr/current/vulkan-resident.md.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T35 — Add cooperative Q8 workgroup reduction (completed)

Both kernels pass 148 FP64-bound workloads including all 44 real matrices, with zero SyncVal errors. Thirteen GPU/policy tests pass; Release build has zero warnings/errors. Three unvalidated process reports show faster small-matrix device times but a regression for 32000-row output; Row64 remains the default. Evidence: docs/benchmarks/q8-cooperative-2026-10-12. No numerical bounds relaxed.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T36 — Keep staging buffers persistently mapped (completed)

Mapped host-visible allocations once for their lifetime; whole-allocation flush/invalidate retained and unmap follows pending-work completion. Nine critical hardware tests pass with SyncVal, including distinct repeated vectors, allocation and concurrent lifetime cases. Build succeeds without warnings. Three timestamp-off whole-model phase reports published in docs/benchmarks/staging-2026-10-12; no mapping-only speedup claimed; noncoherent hardware not exercised.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T37 — Implement resident RMSNorm recording (completed)

Added immutable FP32 pipeline descriptors, owned arena/staging with context lifetime registration, and RMSNorm recording plus synchronous facade. Independent fixture, zeros, profile ranges and RMSNorm-to-Q8 FP64 checks pass with SyncVal; build has zero warnings/errors. Operator tolerance unchanged. docs/vulkan-transformer-operators.md documents scope and lifecycle.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T38 — Implement resident RoPE recording (completed)

Added in-place paired RoPE using the immutable host-computed rotation table. Independent fixture and profile positions 0,1,31,63,127 pass unchanged operator tolerances. Two operator tests pass with zero SyncVal errors; build without warnings/errors. Production recording performs no readback.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T39 — Implement resident SwiGLU recording (completed)

Added sign-stable FP32 SwiGLU and exact gate alias. Independent fixture, extreme signed gates and real feed-forward block pass unchanged tolerances. Three operator tests pass under SyncVal; build without warnings/errors. No activation approximation or intermediate readback in recording.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T40 — Implement resident residual addition (completed)

Elementwise residual recording allows exact aliases and retains explicit barriers. Critical cases cover both exact input aliases, partial-overlap rejection, one-element boundary and normalization chaining. Four operator tests pass with SyncVal; Release build without warnings/errors.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T41 — Implement incremental GPU attention and private KV (completed)

Added stable FP32 causal attention recording and private synchronous attention sessions. Independent Python binary64 fixtures at 1,2,32,64,128 tokens pass unchanged operator bounds. Isolation, reset, full-context rejection, cache budget and context lifetime pass. Five operator/attention tests pass with zero SyncVal errors; build without warnings/errors. No FP64 or subgroup requirement.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T42 — Compose complete resident transformer layers (completed)

Added shared compressed weights/tables, private KV/activation arenas and immutable per-session matrix descriptors with offset-aware Q8 shaders. Complete layers record without intermediate host transfer. Layer 0 at three independent prefixes and all six fixed-input layer boundaries pass the model gates, with one submission/fence, seven MatVecs and 2304 transfer bytes. SyncVal zero errors; build without warnings. docs/vulkan-resident-layers.md records buffers and ownership.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T43 — Integrate single-submission resident model sessions (completed)

Integrated a backend-independent session factory into Formats and complete resident execution into explicit Vulkan runtime/generation. Productive append transfers 129152 bytes, records 43 MatVecs and submits/waits once. Fifteen independent states/logits, official greedy sequences, streaming, 128-token context, reset, session isolation, zero hot allocations and concurrent disposal pass. All 22 GPU/policy tests pass with zero SyncVal errors; all 116 CPU tests pass separately; build has no warnings/errors. Diagnostic captures are excluded from productive counters. docs/stories15m-vulkan-resident.md documents ownership and use.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## T44 — Select CPU legacy and resident paths by measured latency (completed)

Auto now validates and measures all three whole-model paths with timestamps/validation disabled; the graph/layout/operator protocol participates in cache identity. Eight process cache cases pass, including old/missing/invalid resident data and both synthetic cached GPU branches. Three independent short/long-context reports show CPU 4.090–5.754 ms, legacy 142.004–193.679 ms, resident 60.457–80.819 ms per append. CPU remains selected. Three critical policy tests pass; build without warnings/errors. docs/stories15m-planner.md records the exact governing workload and limitations.

Skills: phase-development-project-conventions, phase-development-task-execution, base-dossier-writing; base-project-decision-log for architectural contracts, capabilities-dev-problem-solving-methodology for numerical/performance evidence. Standing authorization includes critical tests and one commit per completed task.

## Model-specific documentation cleanup — 2026-10-13

Removed sixteen model-specific guides, stale optimization narratives and fixture prose. README, TASKS and ProjectInfo now direct development to the multi-model requirements. Removed links to deleted guides; numerical fixtures, raw measurements and production code were preserved. Documentation-only verification: local Markdown link scan; no runtime tests added or run. Skills: phase-development-project-conventions and base-dossier-writing.

## Local quantized-model and kernel scouting — 2026-10-13

Inspected DS4 small, AirLLM-plus and AirLLM at verified revisions, plus official HF artifacts/configs for seven candidate model repositories. Published exact file sizes/hashes, estimated KV budgets and architecture/format gaps. Recommended a candidate portfolio rather than a model-named runtime. DS4 supplies quantized-kernel/residency ideas; AirLLM supplies layer streaming; AirLLM-plus uses mock compute and is not inference performance evidence. No model weights downloaded or external code executed. JSON and local links verified; no runtime tests needed for this research-only task. Skills: phase-scouting-solution-scouting, phase-scouting-technical-feasibility and base-dossier-writing.

## Q4_K_M and UD-IQ scouting — 2026-10-13

Evaluated mixed quantization recipes, verified Unsloth artifact revisions/sizes and documented codec, CPU/Vulkan and quality gaps. Proposed Q4_K/Q6_K first, then an IQ3 comparison on the same model; IQ2 for capacity and IQ1 exploratory. No weights downloaded, binary tensor inventory or performance claims made. Evidence JSON parsed; documentation whitespace checked. Skills: phase-scouting-solution-scouting and base-dossier-writing.

## Repository ideas and APEX/Bonsai applicability — 2026-10-13

Consolidated applicability of DS4, AirLLM and AirLLM-plus for bounded residency, layer/expert streaming and compressed kernels. Added APEX as a MoE streaming/cache candidate and Bonsai PTQ1_0/PQ2_0 as distinct ternary/rotation paths; verified artifact metadata and Prism codec definitions. No weights downloaded or inference performed. JSON and local links checked; no runtime tests for documentation-only work.

## DeepSeek implementation plan — 2026-10-13

Created DeepSeek_todo.md with forty ordered tasks, contract gates, per-task outputs/checks/exclusions and commit policy. Covers generic CPU inference, K/IQ, BitNet, APEX bounded streaming/expert cache, Bonsai codecs/rotations and measured Vulkan. Critical-only testing; performance acceptance requires declared targets. Documentation links and whitespace verified; no runtime tests needed. Skill: analysis-piano-task.

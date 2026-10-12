# Metodo del laboratorio numerico

Corpus v1: xorshift32 con seed 1729, conversione a 24 bit definita nel generatore. I byte non dipendono dall'implementazione di `System.Random`. Nove famiglie critiche per ciascuno dei tre block size; nove combinazioni aggiuntive per le matrici di benchmark 1024², 2048², 4096². Il confronto CPU verifica entrambe le implementazioni: 72 risultati numerici per esecuzione. I test SIMD aggiungono dimensioni che attraversano tutte le code da uno a sette elementi.

## Errori distinti

- `WeightQuantizationRms`: differenza tra pesi FP32 originali e pesi ricostruiti.
- `OutputQuantizationRms`: differenza tra le due MatVec FP64, rispettivamente sui pesi originali e ricostruiti.
- `KernelRms` e `KernelMaxAbsoluteError`: differenza tra kernel FP32 e reference FP64 sui medesimi pesi ricostruiti.

Per ogni riga, il limite del kernel è `gamma(n) × sum(abs(wq × x)) + (columns + blocks) × float.Epsilon`, dove `n = blockSize + blocks + 2` e `gamma(n) = n × 2^-24 / (1 - n × 2^-24)`. È un limite conservativo per questo percorso di accumulo, in assenza di overflow intermedio. I casi con scala massima usano attivazioni piccole per mantenere i risultati finiti. Non è una garanzia per input arbitrari che causano overflow FP32. Il rapporto errore/limite deve essere ≤ 1 per ogni riga, con risultato finito.

## Prestazioni

BenchmarkDotNet 0.14.0, toolchain in-process, job breve con tre warm-up e tre iterazioni misurate, tempo target 200 ms per iterazione. La toolchain evita la creazione di eseguibili temporanei. Il setup genera e quantizza una matrice per caso fuori dal tempo misurato; gli array originali non vengono trattenuti dal benchmark. Il kernel elabora il payload compatto con input e output preallocati.

Il JSON riporta latenza media, deviazione standard, operazioni/s, byte/peso, banda effettiva dei pesi e allocazioni gestite per operazione. La banda è `packedBytes / elapsedSeconds / 10^9`; non include letture ripetute delle attivazioni, traffico cache o scritture output. Il report BenchmarkDotNet conserva hardware e intervalli di confidenza.

Le misure sono sintetiche e a caldo. Cache, frequenza CPU, scheduling e temperatura influenzano i risultati. Non misurano cold start, picco di memoria del processo, energia o token/s; non dimostrano qualità di un modello. Il job breve è adatto alla baseline iniziale, non a dichiarare piccoli vantaggi fra kernel. La verifica delle allocazioni con contatore del thread nei test critici è distinta dalla diagnosi BenchmarkDotNet.

## Confronto CPU

`Scalar` usa sempre il kernel scalar. `Avx2OrScalar` usa AVX2 se il runtime lo abilita, altrimenti il kernel scalar. Entrambi elaborano gli stessi pesi quantizzati con il medesimo seed; non confrontano matrici quantizzate diversamente. Il kernel AVX2 decodifica otto pesi da due byte, applica maschere per zero e segno alle attivazioni, accumula otto lane FP32 e riduce ogni blocco; gli ultimi elementi vengono elaborati scalar. Il diverso ordine di accumulo può cambiare gli ultimi bit, quindi la correttezza si verifica rispetto alla reference FP64 e al limite numerico, non tramite uguaglianza bit per bit tra i kernel.

Per provare il fallback su una CPU AVX2, eseguire i test con `DOTNET_EnableAVX2=0`. La disabilitazione riguarda soltanto quel processo. Il benchmark esporta i nomi dei metodi in `performance.json` e la capacità reale in `environment.json`; i risultati numerici identificano il kernel effettivo. Ogni nuova esecuzione sovrascrive i report rigenerabili: salvare lo snapshot prima della successiva.

## Mapping: avvio e regime

L'opzione `--mapped` confronta `OwnedAuto` e `MappedAuto` su matrici 1024², 2048², 4096² con blocchi da 64. Il job misura il regime dopo warm-up; apertura, validazione e scrittura del file sono escluse. Il percorso mapped include acquisizione/rilascio protetto del mapping per ogni MatVec. Entrambi i percorsi usano gli stessi kernel e gli stessi dati.

`startup.json` riporta separatamente tre prove per dimensione: apertura più validazione completa del payload, primo calcolo successivo, identità con il risultato in memoria. Il codice numerico viene scaldato prima con la matrice in memoria, ma il primo accesso mapped può ancora includere inizializzazioni della callback. Queste prove sono misure di ciclo di vita, non benchmark statistici di SSD.

I file vengono scritti poco prima delle misure, senza espulsione della page cache. Inoltre la validazione iniziale tocca il payload: il primo calcolo non è un accesso a storage freddo. Non si attribuisce quindi al mapping una velocità di avvio a freddo dimostrata. Fixture e file di benchmark vengono creati in directory uniche sotto `.temp/t10/` e rimossi dai loro proprietari.

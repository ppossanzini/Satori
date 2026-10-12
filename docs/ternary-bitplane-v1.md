# TernaryBitplane v1

Formato interno CPU, indipendente da GGUF. Stesse forme, scale e pesi logici del [Ternary2 v1](ternary2-v1.md); nessuna nuova regola di quantizzazione. Blocchi di 64, 128 o 256 elementi, locali alla riga; ultima riga/blocco con padding a zero. Dimensioni positive; payload entro Int32 con aritmetica controllata.

## Blocco

| Regione | Offset | Byte |
|---|---:|---:|
| Scala FP32 little-endian | 0 | 4 |
| Piano positivo P | 4 | blockSize/8 |
| Piano negativo N | 4+blockSize/8 | blockSize/8 |

Il peso logico i usa bit `i%8` del byte `i/8`, meno significativo per primo. `(P,N)`: (0,0) zero, (1,0) +1, (0,1) −1. (1,1) invalido. Peso ricostruito `scale × (P−N)`.

Scala finita, non negativa, escluso negative zero. Scala zero: entrambi i piani interamente zero. Il padding non contiene bit attivi. Payload per blocco `4+blockSize/4` byte, identico al Ternary2; il layout cambia la decodifica, non il numero di byte.

Esempio blockSize 64, scala 0,5, primi otto codici `[+1,0,−1,+1,−1,0,0,+1]`: scala `00 00 00 3F`, P primo byte `89`, N primo byte `14`; tutti gli altri byte zero. Risultato con input uno: 0,5. Il Ternary2 equivalente inizia con payload codici `61 42` dopo la stessa scala.

## Conversione senza perdita

Da Ternary2 già validato: copiare i quattro byte della scala senza ricalcolarla; trasformare codice 01 in P, 10 in N, 00 in nessun bit. Nessun cambiamento di ordine, blocco o padding. Conversione inversa deterministica: P→01, N→10. Round trip byte-identico al Ternary2 originale, compresi FP32 subnormal e blocchi parziali.

Non si quantizza nuovamente dal valore ricostruito: la soglia originaria potrebbe produrre altri codici. Le conversioni possono allocare; la MatVec su oggetto già costruito non alloca.

## Contenitore STB1

Header 32 byte, analogo al contenitore STQ2 ma distinto:

| Offset | Campo |
|---:|---|
| 0 | Magic ASCII `STB1` |
| 4 | Versione UInt16 LE = 1 |
| 6 | Header size UInt16 LE = 32 |
| 8 / 12 / 16 | Righe / colonne / block size Int32 LE |
| 20 | Riservato UInt32 = 0 |
| 24 | Lunghezza payload Int64 LE |

File esatto: header + payload. Nessun checksum. Save crea un file nuovo; una scrittura fallita può lasciare un file incompleto. Import copia e valida; load valida header e payload prima di pubblicare l'oggetto. Mapping del nuovo contenitore non incluso: il confronto fra layout usa entrambi i payload posseduti in memoria, per isolare il costo di decodifica.

## Numerica e benchmark

MatVec scalar: accumulo FP32 dei contributi con segno all'interno del blocco, moltiplicazione per scala e accumulo dei blocchi. AVX2: otto attivazioni per volta, maschere dai piani, coda scalar senza letture fuori input. Nessuna dipendenza da FMA o AVX-512; fallback scalar.

Tolleranza FP64: stessa funzione e stesso corpus del Ternary2. Scalar dei due layout segue lo stesso ordine delle colonne e deve coincidere; SIMD può differire entro il limite documentato nel metodo benchmark. Non si promette aritmetica finita se l'accumulo FP32 va in overflow.

Benchmark T17 misura MatVec scalar completa, inclusa decodifica, su matrici 1024/2048/4096 e block size 64/128/256. T18 aggiunge entrambi i kernel SIMD. Scale, codici, input, hardware, processo, modalità calda e numero di thread identici. Conversione e validazione fuori dal tempo. Riportare anche zero allocazioni e byte/peso; nessuna misura del solo unpack usata come velocità della MatVec.

L'implementazione AVX2 può essere esposta esplicitamente; selezione automatica solo dopo correttezza e vantaggio misurato. Un kernel senza vantaggio resta sperimentale. Nessuna promessa su token/s, qualità o RAM di un modello.

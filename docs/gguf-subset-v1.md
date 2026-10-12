# Profilo GGUF Satori v1

Stato: lettore, adapter numerico e kernel scalar/AVX2 Q8_0/Q4_0 implementati.

## Confine

- GGUF versione 3, little-endian, singolo file immutabile.
- Catalogo di metadati e tensori su mapping read-only.
- Tipi tensoriali ammessi: F32, F16, Q4_0, Q8_0.
- MatVec iniziali: tensori Q8_0 e Q4_0 a due dimensioni.
- Esclusi: altre versioni, big-endian, shard, adapter, conversione ternary automatica, tokenizer, inferenza e GPU.

Un file accettato dal catalogo non equivale a un modello eseguibile. Le restrizioni di questo profilo non definiscono la validità di ogni GGUF.

## Riferimenti fissati

Revisione GGML: `3612c0af93f2cfc8f048295300a61187723b12ae`, verificata il 12 ottobre 2026.

- [Struttura GGUF](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md).
- [Layout dei blocchi](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/ggml-common.h).
- [Decodifica di riferimento](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/ggml-quants.c).
- [Lettore GGML](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/src/gguf.cpp).
- [Versione della quantizzazione](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/include/ggml.h).
- [Tipi dei metadati split nel writer llama.cpp](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/gguf-py/gguf/gguf_writer.py), revisione `efe3626b119ff81033e309c3ba07a770bef2c546`.

## Struttura del contenitore

Header di 24 byte: magic ASCII `GGUF`, versione UInt32, numero tensori UInt64, numero metadati UInt64. Seguono metadati, directory tensoriale, padding e dati.

`dataStart = alignUp(directoryEnd, alignment)`. Gli offset nella directory sono relativi a `dataStart`. L'offset assoluto è la somma controllata dei due valori.

Ogni voce contiene nome con prefisso UInt64 in byte, rango UInt32, dimensioni UInt64, tipo UInt32 e offset UInt64. Rango ammesso: 1–4. Dimensioni positive, rappresentabili come Int64. Il prodotto delle dimensioni e la lunghezza del tensore devono rimanere entro Int64.

Nomi univoci, UTF-8 valido, senza NUL, da 1 a 63 byte. Il limite 63 segue il lettore GGML fissato; il testo della specifica indica 64.

Allineamento predefinito: 32 byte. `general.alignment`, se presente, deve essere UInt32 e una potenza di due tra 8 e 4096. La specifica descrive multipli di otto; questo profilo adotta il vincolo più stretto del lettore GGML e aggiunge il limite 4096.

Layout ammesso: directory in ordine fisico, primo offset relativo zero, ogni offset successivo uguale al precedente più la lunghezza arrotondata all'allineamento. Il file deve contenere anche il padding finale dell'ultimo tensore. Sono rifiutati gap aggiuntivi, alias, sovrapposizioni e ordine fisico diverso dalla directory. Eventuali byte dopo l'ultima regione allineata sono ignorati, mai interpretati come tensori. Il padding non viene usato per calcolare i pesi; il suo contenuto non viene validato.

## Metadati

Tipi scalar ammessi, con gli ID GGUF:

| ID | Tipo | Byte |
|---:|---|---:|
| 0 / 1 | UInt8 / Int8 | 1 |
| 2 / 3 | UInt16 / Int16 | 2 |
| 4 / 5 | UInt32 / Int32 | 4 |
| 6 | Float32 | 4 |
| 7 | Bool | 1 |
| 8 | String UTF-8 con lunghezza UInt64 | variabile |
| 10 / 11 | UInt64 / Int64 | 8 |
| 12 | Float64 | 8 |

Array, ID 9: tipo degli elementi UInt32, numero elementi UInt64, valori consecutivi. Ammessi elementi scalar della tabella, comprese stringhe. Array vuoti ammessi; array annidati esclusi dal profilo. Bool valido soltanto con byte 0 o 1. Valori Float32/Float64 non finiti sono conservati come metadati grezzi; non sono interpretati come parametri di inferenza.

Chiavi univoche, ASCII, da 1 a 65535 byte; grammatica `[a-z0-9_]+(\.[a-z0-9_]+)*`. Valori stringa UTF-8 rigoroso, senza terminatore obbligatorio. NUL consentito nei valori, escluso nelle chiavi e nei nomi tensoriali. Non normalizzare Unicode né cambiare maiuscole o contenuto.

Le chiavi sconosciute con tipo noto sono indicizzate. Un tipo di valore sconosciuto non può essere saltato senza conoscerne la lunghezza: il file viene rifiutato.

### Chiavi interpretate dal lettore

| Chiave | Regola Satori |
|---|---|
| `general.architecture` | String non vuota `[a-z0-9]+`, obbligatoria; identifica il file, non abilita un'architettura |
| `general.type` | String se presente, soltanto `model`; altre varianti fuori profilo |
| `general.quantization_version` | UInt32, obbligatoria con Q4_0/Q8_0; valore ammesso 2 |
| `general.alignment` | Regola della sezione contenitore |
| `general.file_type` | UInt32 se presente; informativo, mai sostitutivo del tipo di ogni tensore |
| `split.count` | UInt16 se presente, soltanto 1 |
| `split.no` | UInt16 se presente, soltanto 0 |
| `split.tensors.count` | Int32 non negativo se presente, uguale al numero dei tensori locali |

Le chiavi `split.*` parziali sono ammesse se rispettano i vincoli. Altre chiavi di modello o tokenizer sono conservate; la loro interpretazione appartiene al profilo del modello.

## Tipi tensoriali

| Tipo | ID | Elementi/blocco | Byte/blocco | Byte/peso |
|---|---:|---:|---:|---:|
| F32 | 0 | 1 | 4 | 4 |
| F16 | 1 | 1 | 2 | 2 |
| Q4_0 | 2 | 32 | 18 | 0,5625 |
| Q8_0 | 8 | 32 | 34 | 1,0625 |

Lunghezza: `product(dimensions) / elementsPerBlock × bytesPerBlock`, con overflow controllato. Per Q4_0/Q8_0 anche la prima dimensione deve essere multipla di 32: i blocchi non attraversano righe. Non aggiungere padding logico come nel Ternary2 interno.

Qualsiasi tipo tensoriale fuori tabella rende il file non supportato, anche se il tensore non viene selezionato. Questo consente di verificare l'intera regione dati senza stimare lunghezze sconosciute. Un file Q4_K_M con Q4_K/Q6_K non rientra nel profilo.

### Contratto numerico Q4_0/Q8_0

Primi due byte del blocco: scala FP16 little-endian, convertita a FP32.

- Q4_0: seguono 16 byte. Per `j = 0..15`, il nibble basso produce `w[j] = d × ((b[j] & 15) - 8)`; quello alto produce `w[j+16] = d × ((b[j] >> 4) - 8)`.
- Q8_0: seguono 32 Int8. `w[j] = d × q[j]`; l'intero intervallo Int8 è rappresentabile.

Le scale possono essere negative, zero o negative zero. Non applicare le regole di validazione Ternary2. T12 valida contenitore e intervalli, senza scansionare numericamente i pesi. T13/T14 verificano una volta le scale del tensore selezionato e rifiutano NaN/Infinity prima dell'esecuzione. F32/F16 sono catalogati; nessuna MatVec specifica viene promessa da T12.

Per una matrice a rango 2: `columns = dimensions[0]`, `rows = dimensions[1]`. La prima dimensione è contigua. Attivazioni e accumuli iniziali FP32; reference numerica FP64 sugli stessi pesi decodificati. L'ordine Q4_0 non è quello di quattro pesi consecutivi a due bit del Ternary2.

## Limiti del parser

Limiti Satori fissi per la prima versione; non sono limiti dello standard.

| Risorsa | Massimo |
|---|---:|
| File | 1 TiB, processo a 64 bit |
| Tensori | 100000 |
| Coppie di metadati | 100000 |
| Stringa di valore | 16 MiB di UTF-8 |
| Elementi per array | 1048576 |
| Header + metadati + directory + padding iniziale | 256 MiB |

Numero tensori zero escluso: i file solo vocabolario non servono al laboratorio. Nessuna allocazione proporzionale a contatori non ancora verificati. Prima di ogni lettura o skip verificare il limite logico e i byte residui. Tutti i calcoli di prodotto, somma, padding e conversione UInt64→Int64 sono controllati.

Il catalogo usa offset e lunghezze Int64. Le finestre esposte alla CPU hanno lunghezza Int32. Un tensore maggiore di `int.MaxValue` byte può essere indicizzato e letto per finestre; le prime MatVec T13/T14 possono rifiutarlo come operazione non supportata.

## Risultato di T12

Modulo: `src/Satori.Formats.Gguf`, dipendente da `Satori.Core` per il mapping. `Satori.Core` non dipende dal modulo GGUF. I futuri kernel numerici ricevono payload e forma; l'adapter GGUF seleziona il tipo senza introdurre GGUF nei kernel.

- Un proprietario del mapping con catalogo immutabile dei tensori.
- Metadati indicizzati per chiave, tipo, posizione e lunghezza; array non materializzati automaticamente.
- Lettura typed degli scalar e delle stringhe su richiesta; accesso scoped ai byte degli array con tipo e conteggio dichiarati.
- Accesso ai pesi per finestre protette dal lifetime di `MappedWeights`.
- Nessuna copia completa del file, nessuna dequantizzazione all'apertura.

L'accesso a un tensore specifica offset relativo e lunghezza della finestra. Il lettore controlla il confine del tensore prima di delegare al mapping. Una finestra non può leggere il padding o i dati del tensore successivo.

File malformato o troncato: `InvalidDataException`, con posizione e contesto. Versione, endian, tipo o variante esclusi: `NotSupportedException`. Superamento dei limiti Satori: `NotSupportedException`, con nome del limite. Errori del filesystem: eccezioni I/O originali. Accesso dopo chiusura: `ObjectDisposedException`. Un'apertura fallita rilascia tutte le risorse. Nessun catalogo parziale pubblicato.

## Verifica prevista

Il [catalogo fixture](gguf-fixtures-v1.md) fissa casi validi, fuori profilo e corrotti. Il lettore è verificato sulle golden, su contenitori corrotti e grandi, e sul corpus pubblico stories15M. Le MatVec sono confrontate con decodifica FP64 indipendente; evidenze in `docs/benchmarks/gguf-2026-10-12/`.

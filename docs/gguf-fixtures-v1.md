# Fixture GGUF v1

Profilo: [gguf-subset-v1.md](gguf-subset-v1.md). Le fixture descrivono contenitori, non modelli LLM completi.

## Golden disponibili

Directory: `fixtures/gguf/v1/`. Il manifest contiene SHA-256, offset, forme, lunghezze e valori ricostruiti attesi. Il generatore Python usa soltanto `struct` e non richiama codice Satori.

| File | Tipo e forma GGML | Blocco o valori | Risultato con input di uno |
|---|---|---|---:|
| `q4_0.gguf` | Q4_0, `[32, 1]` | Scala 0,5; 16 byte `F0`: 16 pesi −4, poi 16 pesi 3,5 | −8 |
| `q8_0.gguf` | Q8_0, `[32, 1]` | Scala 0,5; 16 Int8 −8, poi 16 Int8 7 | −8 |
| `f32.gguf` | F32, `[3]` | 1, 2, 3 | 6, solo dato reference |
| `f16.gguf` | F16, `[3]` | 0, 1, −1 | 0, solo dato reference |

Q4_0 e Q8_0 rappresentano gli stessi pesi. Il risultato atteso è calcolabile a mano; evita che codec e kernel condividano un errore non rilevato. Gli header usano GGUF v3, allineamento predefinito, una voce tensoriale e `general.architecture=llama`. I file quantizzati dichiarano quantization version 2.

## Casi da implementare in T12

La tabella resta il catalogo di accettazione. Le verifiche implementate coprono golden, troncamenti, metadati scalar/array, UTF-8, allineamenti, dimensioni/overflow, overlap, finestre e disposal; il corpus pubblico verifica anche cataloghi misti. Non ogni combinazione della tabella è un test separato. I file grandi e i casi corrotti vengono generati sotto `.temp/`, senza aggiungere modelli al repository.

| ID | Caso | Esito del lettore |
|---|---|---|
| V01 | Quattro golden disponibili | Catalogo uguale al manifest; byte identici |
| V02 | File misto F32/F16/Q4_0/Q8_0; offset cumulativi allineati | Tutte le voci indicizzate; nessuna dequantizzazione |
| V03 | Allineamento esplicito 8, 64 o 4096 | `dataStart` e offset assoluti corretti |
| V04 | Tutti gli scalar e array piatti; array vuoti e stringhe multibyte | Tipi, conteggi, valori e confini corretti |
| V05 | Valore stringa con NUL, chiavi sconosciute, metadati in ordine diverso | Contenuto conservato |
| V06 | Rango 1, 3, 4 con tipi ammessi | Catalogati; nessuna MatVec implicita |
| V07 | File sparse con secondo tensore oltre 2 GiB, primo tensore F32 grande e offset canonici | Offset Int64 e lettura per finestre senza allocazione completa |
| V08 | Payload con scala negativa, zero, negative zero o non finita | Catalogato senza validazione numerica; verifica delle scale rinviata al kernel |
| V09 | Byte aggiuntivi dopo il padding dell'ultimo tensore | Ignorati |
| U01 | Versione 1, 2 o maggiore di 3; header big-endian riconoscibile | `NotSupportedException` |
| U02 | Tipo tensoriale non ammesso, anche se non selezionato | `NotSupportedException` |
| U03 | Array annidato o tipo metadata sconosciuto | `NotSupportedException` |
| U04 | Shard dichiarato, quantization version diversa da 2, `general.type` diverso da `model` | `NotSupportedException` |
| U05 | Allineamento potenza di due oltre 4096, nome di 64 byte, rango oltre 4 | `NotSupportedException` |
| U06 | Numero tensori zero o limite Satori superato | `NotSupportedException`, limite identificato |
| U07 | Gap o directory fuori ordine con intervalli altrimenti validi e non sovrapposti | `NotSupportedException`, layout fuori profilo |
| E01 | Magic errato; header troncato | `InvalidDataException` |
| E02 | Contatore, stringa o array dichiarato oltre i byte residui | `InvalidDataException`, senza allocazione proporzionale |
| E03 | Tipo errato per una chiave interpretata; metadati obbligatori mancanti | `InvalidDataException` |
| E04 | Bool diverso da 0/1, UTF-8 invalido, chiave illegale, NUL in nome | `InvalidDataException` |
| E05 | Chiave o nome tensoriale duplicato, nome vuoto | `InvalidDataException` |
| E06 | Rango zero, dimensione zero, UInt64 oltre Int64 o prodotti in overflow | `InvalidDataException` |
| E07 | Prima dimensione Q4_0/Q8_0 non divisibile per 32 | `InvalidDataException` |
| E08 | Allineamento zero o non potenza di due, offset non allineato | `InvalidDataException` |
| E09 | Somma offset/lunghezza in overflow, payload o padding finale fuori file | `InvalidDataException` |
| E10 | Intervalli tensoriali sovrapposti | `InvalidDataException` |
| L01 | Apertura fallita in ogni sezione; eccezione nella callback | Risorse rilasciate; nessun catalogo parziale |
| L02 | Lettura concorrente con Dispose, nuova lettura dopo Dispose | Accesso attivo valido; nuovo accesso rifiutato |
| L03 | Finestra dentro il file ma oltre il tensore selezionato | Rifiuto prima della callback; nessuna lettura di padding o tensore adiacente |

Quando un input viola più regole, prevalgono bounds/overflow e struttura prima dei controlli di profilo. In E02, un conteggio già oltre un limite Satori può essere rifiutato come U06 prima di leggere il resto del file.

## Casi numerici per T13/T14

- Golden Q4_0/Q8_0: decodifica per posizione e MatVec attesa −8.
- Scale positive/negative, ±0 e FP16 subnormal; NaN/Infinity rifiutati all'apertura del tensore numerico.
- Q8_0 con Int8 −128 e 127; Q4_0 con tutti i nibble 0–15 nelle due metà.
- Matrici non quadrate, più righe e più blocchi per riga.
- Confronto FP64 sugli stessi pesi decodificati; nessuna confusione con errore di quantizzazione.
- Identità mapped/owned e nessuna allocazione nel percorso caldo.

Le quattro golden sono state lette dal reader ufficiale gguf-py alla revisione fissata. I cataloghi dei due file pubblici stories15M (57 tensori ciascuno) sono confrontati voce per voce con lo stesso reader. La generazione delle reference numeriche rimane indipendente da Satori.

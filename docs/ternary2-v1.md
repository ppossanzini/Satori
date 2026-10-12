# Ternary2 v1

Formato sperimentale interno, distinto dai formati GGUF. Matrice row-major con righe positive e colonne positive. Blocchi di 64, 128 o 256 pesi, senza attraversare le righe. L'ultimo blocco di ogni riga è completato con zeri.

Ogni blocco occupa `4 + blockSize / 4` byte: scala FP32 IEEE754 little-endian, poi pesi a due bit. Nel byte, il primo peso occupa i bit 0–1; seguono 2–3, 4–5, 6–7. `00 = 0`, `01 = +1`, `10 = -1`, `11` invalido. Peso ricostruito: codice con segno × scala. Scala finita non negativa; blocchi nulli hanno scala +0 e payload nullo. Padding sempre nullo. Con scala zero anche tutti i pesi logici devono essere nulli.

Quantizzazione deterministica: scala = media dei valori assoluti dei soli pesi validi del blocco, accumulata FP64 e convertita FP32. Se scala zero, il blocco è nullo. Altrimenti scegliere +1 per `x > scala / 2`, −1 per `x < −scala / 2`, zero negli altri casi, inclusi i pareggi. Soglia calcolata in FP64 per evitare underflow sui subnormal. Non è una quantizzazione calibrata per qualità LLM.

Input non finiti rifiutati. Dimensioni e lunghezze calcolate con overflow controllato; ogni payload di matrice ammette al massimo `int.MaxValue` byte. L'importazione in memoria copia e valida il payload prima di renderlo disponibile ai kernel. Il payload grezzo non contiene un header: forma e block size sono argomenti espliciti. Il contenitore di file STQ2 descritto sotto aggiunge questi metadati.

Esempio: blocco da 64, primi quattro valori `[2, -2, 0, 0]`, altri zero. Scala = 0,0625; primi quattro byte `00 00 80 3D`, primo byte pesi `09`, rimanenti 15 byte pesi nulli. Un blocco tutto zero occupa 20 byte nulli.

MatVec: attivazioni FP32, somma FP32 dei contributi con segno per blocco, moltiplicazione per scala, accumulo FP32 tra blocchi. Nessuna espansione completa o allocazione nel kernel. L'errore di accumulo va confrontato con FP64 sugli stessi pesi ricostruiti; l'errore di quantizzazione va misurato separatamente rispetto ai pesi originali.

## Contenitore STQ2 v1

File little-endian: header fisso di 32 byte, seguito dal payload Ternary2 v1. Nessun dato finale aggiuntivo ammesso.

| Offset | Tipo | Contenuto |
|---:|---|---|
| 0 | 4 byte ASCII | `STQ2` |
| 4 | UInt16 | Versione: 1 |
| 6 | UInt16 | Lunghezza header: 32 |
| 8 | Int32 | Righe positive |
| 12 | Int32 | Colonne positive |
| 16 | Int32 | Block size: 64, 128 o 256 |
| 20 | UInt32 | Riservato: zero |
| 24 | Int64 | Lunghezza payload, uguale a quella derivata dalla forma |

`Ternary2Matrix.Save` crea un nuovo file senza sovrascrivere file esistenti. Un errore I/O durante la scrittura può lasciare un file parziale; il caricatore rifiuta header/payload troncati. Non è un salvataggio atomico né un formato GGUF.

`MappedTernary2Matrix` valida l'header e tutto il payload una volta all'apertura, senza copiarlo in un array. Ogni MatVec acquisisce il mapping per la durata dell'intero calcolo e richiama gli stessi kernel scalar/AVX2 usati dalla matrice in memoria. La chiusura non invalida un calcolo già avviato, ma impedisce quelli successivi.

Il proprietario generico supporta file con offset oltre 2 GiB; il contenitore attuale è limitato a un singolo payload di matrice di massimo `int.MaxValue` byte. Non sono ancora supportati tensori singoli più grandi, collezioni di tensori o checksum. Il file deve rimanere immutabile dopo la validazione e fino al termine degli accessi attivi.

# Forward CPU Stories15M

`StoriesModel` apre soltanto l'artefatto Q8_0 del [profilo fissato](stories15m-profile-v1.md). Controlla dimensione, SHA-256, metadata e catalogo dei 57 tensori prima dell'esecuzione. Possiede il file mappato; le norme F32 sono piccole copie validate. Le matrici restano compresse.

`Forward(tokens, logits, scalar, observer)` ricalcola il prefisso e scrive i 32000 logits dell'ultimo token. Il prefisso contiene da 1 a 128 ID validi, BOS incluso quando richiesto dal tokenizer. Il chiamante possiede logits e token. I buffer devono essere disgiunti. Gli errori su dimensioni e ID vengono rilevati prima di scrivere logits.

Ogni chiamata crea K/V temporanei FP32 per il prefisso, senza conservare stato tra chiamate. La cache incrementale appartiene al task successivo. Le chiamate e la chiusura dello stesso modello sono serializzate. I pesi devono restare immutabili sul filesystem durante la vita del modello.

L'observer opzionale riceve 15 stati diagnostici dell'ultimo token: otto stati del primo layer, l'output di ciascuno dei sei layer e la normalizzazione finale. Gli span sono transitori; conservarli richiede una copia. Se il callback fallisce, l'eccezione raggiunge il chiamante e una chiamata successiva ricomincia senza stato residuo.

```csharp
using var model = new StoriesModel(path);
float[] logits = new float[StoriesModel.VocabularySize];
model.Forward([1, 9038, 2501, 263, 931], logits);
```

Questo percorso non genera token e non stampa una continuazione. Il tokenizer è utilizzabile separatamente per costruire il prefisso.

## Verifica numerica

Ambiente: Linux x64, Intel Core Ultra 7 155H, .NET 10.0.12, build Release. Reference primaria NumPy FP64 sugli stessi pesi Q8_0 ricostruiti, con hash delle fixture verificati durante il confronto.

| Prompt | MatVec | Massimo errore assoluto, stati e logits | Massimo RMS per tensore | Argmax |
|---|---|---:|---:|---:|
| empty | AVX2 | 8.289607e-6 | 2.654648e-6 | 9038 |
| story | AVX2 | 6.962798e-6 | 1.633108e-6 | 29892 |
| italian | AVX2 | 9.378695e-6 | 2.531248e-6 | 450 |
| empty | scalar | 1.430831e-5 | 2.286798e-6 | 9038 |
| story | scalar | 8.812336e-6 | 2.018217e-6 | 29892 |
| italian | scalar | 7.836607e-6 | 2.614642e-6 | 450 |

Tutti i componenti rispettano il gate del profilo, senza allargare le tolleranze. Gli argmax coincidono con il riferimento. La suite comprende nove nuovi casi di integrazione: sei confronti completi, stato/lifetime/errori, contesto massimo e artefatto incompatibile. Suite totale: 88 test passati con AVX2 e 88 con AVX2 disabilitato.

A 128 token il test misura 1782640 byte di allocazioni temporanee dopo warm-up; logits esclusi perché forniti dal chiamante. K/V occupano 1769472 byte; scratch più logits resta sotto il budget 256 KiB. Nessuna dequantizzazione completa. Questi valori non misurano working set, TTFT o velocità di generazione.

## Riproduzione

I test `Category=ModelIntegration` richiedono il corpus scaricato e verificato:

```sh
python tools/download_gguf_corpus.py
dotnet build Satori.sln -c Release --no-restore -m:1
dotnet test tests/Satori.Numerics.Tests -c Release --no-build --no-restore -m:1
DOTNET_EnableAVX2=0 dotnet test tests/Satori.Numerics.Tests -c Release --no-build --no-restore -m:1
```

Per eseguire solo i test portabili senza pesi: aggiungere `--filter 'Category!=ModelIntegration'`. L'assenza dei pesi non viene interpretata come superamento dei test di integrazione. Nessun download avviene dentro i test.

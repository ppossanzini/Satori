# KV cache Stories15M

`StoriesModel.CreateSession(scalar)` crea una sequenza indipendente con contesto fisso 128. `StoriesSession` possiede K/V FP32 e scratch riutilizzabili; prende in prestito il modello. La chiusura della sessione non chiude i pesi. Il modello deve restare aperto durante il calcolo.

| Operazione | Contratto |
|---|---|
| `Count` | Token elaborati, da 0 a 128, BOS incluso |
| `TryAppend(token, logits)` | Calcola il nuovo token e scrive 32000 logits; restituisce true e avanza Count dopo il successo |
| `TryAppend` a Count=128 | Restituisce false; logits e cache restano invariati; nessuna eviction |
| `Reset()` | Azzera K/V, scratch e Count senza riallocare |
| `Dispose()` | Azzera i buffer e rilascia il riferimento al workspace; idempotente |

Token invalidi e output con dimensione diversa da 32000 vengono rifiutati prima di scrivere o avanzare. Dopo la chiusura del modello, un calcolo necessario solleva ObjectDisposedException e non avanza Count. Il reset rimane disponibile perché opera soltanto sui buffer posseduti. Dopo la chiusura della sessione, Count, append e reset sollevano ObjectDisposedException.

Layout: sei regioni per layer, ognuna con 128 posizioni da 288 float. K viene memorizzato dopo RoPE, V senza rotazione. L'attention legge soltanto le posizioni da zero a quella del token corrente. Il token nuovo scrive soltanto la nuova posizione in ciascun layer. Nessun token precedente viene ricalcolato.

Le operazioni della stessa sessione sono serializzate. Sessioni diverse conservano prefissi separati e condividono soltanto i pesi. Il modello serializza i calcoli e il proprio disposal. `StoriesModel.Forward` conserva il ricalcolo completo come riferimento separato.

```csharp
using var model = new StoriesModel(path);
using var session = model.CreateSession();
float[] logits = new float[StoriesModel.VocabularySize];
foreach (int token in promptTokens)
{
    if (!session.TryAppend(token, logits)) break;
}
session.Reset();
```

## Verifica

Build Release senza warning/errori; 97 test critici passati con AVX2 e 97 con AVX2 disabilitato. Nove nuovi casi di integrazione coprono confronto per token sui tre prompt ufficiali in scalar e AVX2, contesto pieno, reset, memoria, errori, lifetime e sessioni concorrenti. I logits incrementali coincidono esattamente con il ricalcolo dello stesso prefisso. Il confronto lungo verifica le posizioni 0, 1, 31, 63 e 127 su una sequenza di 128 token. Dopo un reset da contesto pieno, un prefisso diverso coincide con una chiamata senza stato.

K/V occupa esattamente 1769472 byte, pari a 1,6875 MiB per sessione. Costruzione, scratch incluso: 1782808 byte misurati su Linux x64/.NET 10.0.12, sotto il limite verificato di 2 MiB. Append e reset non allocano memoria gestita dopo il rispettivo warm-up. Logits e pesi sono esclusi dal conteggio della sessione. Non sono misure di working set o prestazioni end-to-end.

Riproduzione e download del corpus seguono [la verifica del forward](stories15m-forward.md). I test sono nella categoria ModelIntegration. Generazione, gestione dell'arresto EOS, sampling e persistenza della cache rimangono fuori da questo contratto.

# Generazione Stories15M

Satori genera completamenti raw da Stories15M Q8_0 su CPU. Il percorso usa tokenizer del profilo, prefill sequenziale, cache incrementale, argmax e decoder UTF-8 della continuazione. Greedy resta il default; sampling configurabile opzionale, senza chat template.

```sh
dotnet run --project src/Satori.Console -c Release --no-build -- generate .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12
```

Output verificato:

```text
, there was a little girl named Lily. She loved
```

La console stampa solo la continuazione su stdout. Il riepilogo su stderr riporta motivo di arresto e conteggi. Il prompt vuoto è ammesso e codificato come BOS. Modello diverso dall'artefatto fissato, limite non valido e prompt oltre contesto producono un messaggio senza stack trace e codice di uscita 1.

## API e opzioni

`StoriesGenerator` possiede modello e tokenizer. Ogni `Generate` crea una sessione indipendente e la rilascia anche in caso di errore del callback. Chiamate e disposal sono serializzati. `onText` riceve solo testo UTF-8 completo disponibile; il risultato contiene la stessa concatenazione, gli ID generati, il numero di token prompt e il motivo di arresto.

| Oggetto opzioni | Campo | Default | Validazione |
|---|---|---:|---|
| StoriesGenerationOptions | MaxNewTokens | 32 | Intero non negativo; limitato dallo spazio di contesto disponibile |
| StoriesGenerationOptions | Scalar | false | Se true, MatVec scalar; altrimenti dispatch CPU automatico |

Le opzioni sono facoltative e non contengono segreti. I default sono dichiarati nel record; la console traduce gli argomenti in questo oggetto. `Generate` valida il budget prima del prefill.

```csharp
using var generator = new StoriesGenerator(path);
var result = generator.Generate("Once upon a time", new(MaxNewTokens: 12), Console.Write);
```

## Arresto e limiti

- Greedy sceglie il massimo logit finito; parità risolta dall'ID minore. Logits vuoti o non finiti vengono rifiutati.
- Budget effettivo: min(MaxNewTokens, 128 - promptTokens), con BOS incluso nel prompt.
- EOS viene incluso negli ID e nel conteggio generato, soppresso dal testo e termina subito la generazione con Eos.
- Esaurimento dello spazio disponibile: ContextLimit. In caso di coincidenza con il limite richiesto prevale ContextLimit.
- Esaurimento del solo limite richiesto: TokenLimit.
- Budget zero: nessun prefill, nessun testo. Un prompt già pieno restituisce ContextLimit.
- Nessuna eviction o estensione del contesto. Il token finale non richiede un nuovo forward quando non servono altri logits.

Una continuazione non ristampa il prompt. Il decoder rimuove al massimo uno spazio iniziale solo quando il token precedente è BOS. Il flush finale rende esplicita una sequenza UTF-8 rimasta incompleta tramite U+FFFD.

## Verifica

I dodici token dei tre casi ufficiali coincidono con gli oracoli fissati, sia scalar sia AVX2. Verificati anche testo completo della continuazione, identità dello streaming, budget predefinito, ripetibilità, budget zero, contesto pieno e quasi pieno, prompt eccessivo, recupero dopo errore callback e lifetime. Gli oracoli di dodici token non contengono EOS: selezione dell'ID e soppressione del pezzo speciale sono coperte separatamente; non attestano un arresto EOS end-to-end su questi prompt.

Le verifiche console includono completamento reale, rifiuto di un artefatto diverso e limite negativo, con codici di uscita controllati. Riproduzione della suite e download dei pesi seguono [la verifica del forward](stories15m-forward.md). TokenLimit e ContextLimit sono stati osservati dalla console. Nessuna misura di velocità o qualità generale è dichiarata qui.

Build Release senza warning/errori. Suite finale: 109 test passati con AVX2 e 109 con AVX2 disabilitato, con PGO predefinito. Il percorso stringhe del parser include un [workaround locale verificato per il JIT](diagnostics/dotnet-string-callback.md).

## Sampling configurabile

```sh
dotnet run --project src/Satori.Console -c Release --no-build -- generate .temp/gguf-corpus/stories15M-q8_0.gguf "Once upon a time" 12 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 123
```

`StoriesGenerationOptions.Sampling` accetta `Satori.Core.SamplingOptions`; null usa i default del record. Temperature deve essere finita e >= 0 (default 0: greedy); TopK >= 0 (default 0: nessun filtro, valori oltre vocabolario limitati alla sua dimensione); TopP finito in (0,1] (default 1); Seed unsigned a 64 bit (default 0). Tutte le opzioni vengono validate anche con budget zero o temperature zero. CLI con numeri invariant e flag successivi al budget opzionale; flag ripetuti usano l'ultimo valore.

Ordine: logit decrescente con parità all'ID minore, softmax con temperature, top-k, top-p sulle probabilità rinormalizzate del top-k. Il nucleus conserva il prefisso minimo con massa >= TopP, sempre almeno un token. Accumuli e pesi FP64; sottrazione del massimo prima della divisione, così temperature estremamente piccole possono annullare pesi secondari senza overflow positivo. I logits originali non vengono modificati. Greedy ignora i filtri dopo la validazione.

Il generatore crea un TokenSampler per chiamata e riparte sempre dal seed indicato. SplitMix64 usa aritmetica unsigned modulo 2^64, incremento 9E3779B97F4A7C15, mixer BF58476D1CE4E5B9 e 94D049BB133111EB; i 53 bit alti diventano un numero in [0,1). Il draw sceglie il primo intervallo con cumulata strettamente maggiore del valore estratto. Il PRNG è fissato; Math.Exp e piccole differenze nei logits possono comunque cambiare un esito vicino a una soglia fra piattaforme diverse. Non è un generatore crittografico.

Il sampler possiede scratch O(vocabolario), ordina O(V log V) per token e non è thread safe. Non sono introdotti obiettivi di throughput: benchmark rinviato a T26. EOS, decoder, budget e contesto mantengono il contratto precedente.

Verifica T25: build Release senza warning/errori; 114 test critici passati con AVX2 e 114 senza AVX2. Cinque nuovi casi coprono vettori SplitMix64 noti, distribuzione contro probabilità analitiche (60000 draw senza filtri e 50000 con nucleus), ordine top-k/top-p, parità, temperature estreme, input non finiti, seed ripartito per chiamata, streaming e identità scalar/AVX2 sul prompt verificato. Soglie statistiche fisse e seed fissato; nessun test probabilisticamente variabile tra run. Console con seed 123 ripetuto: `, there was a little girl named Lily. She had`; sei configurazioni CLI invalide rifiutate con uscita 1. Gli oracoli greedy esistenti restano identici. Le limitazioni EOS degli oracoli precedenti restano valide.

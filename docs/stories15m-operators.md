# Operatori CPU Stories15M

`TransformerOps` fornisce RMSNorm, RoPE a coppie adiacenti con base 10000, SwiGLU, somma residuale e attention causale con uguale numero di teste Q/K/V. La MatVec riutilizza i kernel Q8_0 esistenti. Nessun forward, tokenizer o proprietario della cache viene introdotto qui.

Gli input numerici devono essere finiti e i risultati rappresentabili in FP32. Le dimensioni sono esatte; i buffer sono del chiamante. RMSNorm, SwiGLU e somma ammettono output coincidente con un input; le sovrapposizioni parziali sono respinte prima di scrivere. RoPE modifica il vettore sul posto. L'attention richiede output e scratch disgiunti dagli input.

RMSNorm accumula i quadrati in FP64 per evitare overflow con input FP32 finiti; epsilon deve essere positivo e finito. RoPE usa posizioni da zero e applica la stessa rotazione a ogni testa. SwiGLU calcola `SiLU(gate) * up` con sigmoid stabile anche per gate molto negativi. Attention accumula dot product e softmax in FP64, sottrae il massimo, quindi ricostruisce l'output FP32. Queste sono implementazioni CPU di riferimento, senza promesse di prestazioni SIMD.

K/V sono disposti per token, quindi per testa e componente. Il chiamante passa esattamente il prefisso da posizione 0 a quella corrente; le posizioni future non fanno parte dello span. Lo scratch contiene almeno un double per token, viene riutilizzato per ogni testa e non conserva stato. Allocazione, reset e limite 128 della cache appartengono al task della cache.

`GgufQuantizedMatrix.ReadRow` ricostruisce soltanto la riga Q8_0 selezionata, con controllo dei confini e callback protetta dal lifetime del mapping. Riutilizza la validazione delle scale effettuata dal costruttore. `QuantizedRow.DecodeQ8` richiede payload già validato e buffer disgiunti. `GgufFloatVector` copia una piccola norma rank-1 F32, decodifica little-endian e rifiuta valori non finiti; la copia resta valida dopo la chiusura del file. Non viene espansa alcuna matrice completa.

## Evidenza numerica

`tools/transformer_reference.py` produce `fixtures/operators/transformer-v1.json`: input arrotondati FP32, formule indipendenti NumPy FP64, seed 20019. Copre due teste da 48 componenti, posizione RoPE 37, sette token e gate da -1000 a +1000. Il test confronta ogni componente con tolleranza `2e-6 + 2e-6 * abs(reference)` e richiede output finiti. I test aggiuntivi coprono input FP32 massimi, attention a un solo token, alias, dimensioni, codici Q8 signed, scale FP16 negative, letture F32, disposal e zero allocazioni dopo warm-up.

Riproduzione dalla radice: `python tools/transformer_reference.py`, build Release e suite `Satori.Numerics.Tests`. NumPy serve solo a rigenerare l'oracolo, senza dipendenze nel runtime .NET. Hash e versione del generatore sono in `fixtures/operators/manifest.json`. Le fixture complete del modello restano il gate successivo per il forward.

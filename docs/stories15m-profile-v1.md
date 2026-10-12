# Primo modello: Stories15M Q8_0

Profilo CPU fissato per il primo percorso prompt→testo. Modello base di completamento, dominio TinyStories in inglese. Nessun chat template, ruolo system o messaggio instruct. Le fixture multilingue verificano il tokenizer; non attestano qualità linguistica.

## Identità e licenza

| Campo | Valore |
|---|---|
| Repository GGUF | `ggml-org/tiny-llamas` |
| Revisione | `99dd1a73db5a37100bd4ae633f4cfce6560e1567` |
| File | `stories15M-q8_0.gguf` |
| SHA-256 | `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6` |
| Byte | 26671328 |
| Modello originale | `karpathy/tinyllamas`, Stories15M |
| Revisione della scheda originale | `0bd21da7698eaf29a0d7de3992de8a46ef624add` |
| Licenza dichiarata dall'autore | MIT |

La [scheda originale fissata](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/README.md) dichiara MIT e training sull'architettura Llama 2 per llama2.c. Autore: Andrej Karpathy. La [scheda del repository GGUF](https://huggingface.co/ggml-org/tiny-llamas/blob/99dd1a73db5a37100bd4ae633f4cfce6560e1567/README.md) non aggiunge una licenza distinta. Il repository Satori conserva fixture e provenienza; i pesi si scaricano separatamente con verifica dell'hash.

Il profilo accetta questo artefatto esatto. Hash diverso, metadati incompatibili, tensori mancanti o forme diverse richiedono un nuovo profilo. La versione GGUF e la quantizzazione seguono [GGUF v1](gguf-subset-v1.md).

## Topologia

| Parametro | Valore |
|---|---:|
| Architettura | llama decoder-only |
| Layer | 6 |
| Embedding / hidden | 288 |
| FFN | 768 |
| Teste Q / K / V | 6 / 6 / 6 |
| Dimensione di ogni testa | 48 |
| Vocabolario | 32000 |
| Contesto massimo | 128 token, BOS e token generati inclusi |
| RMSNorm epsilon | 9.999999747378752e-6 |
| RoPE | coppie adiacenti, 48 dimensioni, base 10000, scala 1 |
| Bias | assenti |

Head count KV e base RoPE non sono presenti nel file. I valori sono fissati dai default dell'[implementazione ufficiale](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-model.cpp). Nessun GQA, MoE, sliding window, ALiBi o scaling RoPE.

Directory esatta: 57 tensori. 44 matrici Q8_0; 13 vettori F32. Il catalogo completo con forme e byte è in [reference.json](../fixtures/models/stories15m-q8-v1/numpy/reference.json).

| Tensore | Forma GGML, prima dimensione contigua | Tipo |
|---|---|---|
| `token_embd.weight`, `output.weight` | [288, 32000] | Q8_0 |
| `output_norm.weight` | [288] | F32 |
| `blk.i.attn_q/k/v/output.weight` | [288, 288] | Q8_0 |
| `blk.i.attn_norm.weight`, `blk.i.ffn_norm.weight` | [288] | F32 |
| `blk.i.ffn_gate.weight`, `blk.i.ffn_up.weight` | [288, 768] | Q8_0 |
| `blk.i.ffn_down.weight` | [768, 288] | Q8_0 |

`i=0..5`. Embedding e output hanno payload identici ma regioni distinte. Il forward usa i tensori dichiarati, senza deduzione automatica del tying. Il file contiene 24407712 elementi, inclusa la duplicazione dell'output; il nome 15M si riferisce al modello con pesi condivisi.

## Tokenizer e testo

- `tokenizer.ggml.model=llama`: algoritmo SPM del runtime ufficiale fissato, vocabolario/score/tipi dagli array GGUF.
- 31741 token normali, 256 byte fallback, 2 control, 1 unknown. ID unknown=0, BOS=1, EOS=2; byte 0x00..0xFF → ID 3..258.
- Padding e separator non disponibili: il valore UInt32 4294967295 è un sentinel, non un ID valido.
- Input UTF-8 valido, massimo 16384 byte prima del tokenizer, derivato da UTF-16 rigoroso; surrogate isolate rifiutate. Nessuna normalizzazione Unicode.
- Testo non vuoto: aggiungere uno spazio iniziale e sostituire gli spazi ASCII con U+2581. Tab, newline e spazi ripetuti restano distinti. Testo vuoto: nessun prefisso.
- Merge della coppia presente nel vocabolario con score FP32 maggiore; parità risolta dalla posizione sinistra minore. Ricalcolo delle coppie adiacenti; resegmentazione e byte fallback come nel [tokenizer fissato](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-vocab.cpp).
- Encoding per generazione: `add_special=true`, BOS inserito una volta, EOS non aggiunto, `parse_special=false`. Le stringhe letterali `<s>` e `</s>` sono testo comune.
- Encoding senza special disponibile per la verifica. Empty con special produce [1]; senza special produce [].
- Decodifica di pezzi in byte; i byte UTF-8 parziali si uniscono prima della conversione a testo. I control non vengono stampati. EOS interrompe la generazione.
- Decodifica completa conforme a `llama_detokenize(remove_special=true, unparse_special=false)`. Con BOS può conservare lo spazio di prefisso; le fixture fissano il risultato esatto.
- Streaming di una continuazione: non ristampare il prompt. Se il token precedente è BOS, rimuovere al massimo uno spazio iniziale dal primo pezzo stampato; poi conservare gli spazi.

Tokenizer specifico del profilo; SentencePiece esterno non è una dipendenza runtime. Non è ammesso sostituire l'algoritmo con un BPE generico o normalizzare whitespace per ottenere gli stessi token.

## Operatori necessari

| Operatore | Contratto |
|---|---|
| Lettura F32 | Vettori norm little-endian, finiti, dimensione 288; nessun nuovo formato quantizzato |
| Embedding Q8_0 | Ricostruzione della sola riga selezionata, 288 valori FP32 |
| MatVec Q8_0 | Kernel verificati; attivazioni FP32, nessuna quantizzazione delle attivazioni |
| RMSNorm | `scale * x / sqrt(mean(x*x) + epsilon)` |
| RoPE | Per coppia j, angolo `position * 10000^(-2*j/48)`; ruotare Q e K, posizioni da zero |
| Attention causale | Q·K / sqrt(48), softmax stabile sui soli token 0..position, somma pesata di V per testa |
| KV cache | K dopo RoPE e V senza RoPE, FP32, una sequenza, reset esplicito |
| SiLU/SwiGLU | `SiLU(gate) * up`, con `SiLU(x)=x/(1+exp(-x))` |
| Residual | Somma elemento per elemento dopo attention e dopo FFN |
| Output | RMSNorm finale, MatVec output, 32000 logits FP32 |
| Greedy | Argmax; parità risolta dall'ID più piccolo; EOS o limite token/context |

Per layer: RMSNorm → Q/K/V → RoPE Q/K → attention → proiezione → residual → RMSNorm → gate/up → SwiGLU → down → residual. Dopo il sesto layer: RMSNorm finale → output.

Non manca alcun tipo tensoriale nel catalogo GGUF. T20 deve aggiungere lettura dei vettori F32 e gather della riga Q8_0 insieme agli operatori. Non è richiesta una nuova famiglia di quantizzazione prima di T20.

## Memoria e limiti iniziali

Valori calcolati per il futuro runtime Satori; non sono misure di consumo già ottenute.

| Regione | Budget |
|---|---:|
| Mapping file | 26671328 byte di spazio virtuale; circa 25,44 MiB se tutte le pagine sono residenti |
| KV FP32 | `2*6*128*288*4` = 1769472 byte, 1,6875 MiB |
| Scratch numerico, logits e tabella RoPE | massimo 256 KiB |
| Tokenizer e indice posseduti | massimo 16 MiB |
| Working set totale, inclusa .NET | obiettivo ≤256 MiB, da misurare in T26 |

Nessuna copia completa o dequantizzazione permanente delle matrici. La memoria mappata non garantisce che tutte le pagine restino in RAM. Non dedurre il working set dal solo spazio virtuale.

Contesto fisso 128, una sequenza, prefill iniziale sequenziale. Default generazione: greedy, massimo 32 nuovi token; limite effettivo `min(maxNewTokens,128-promptTokens)`. Prompt oltre 128 token rifiutato prima del forward. Contesto pieno restituisce motivo di arresto `ContextLimit`; nessuna estensione o eviction implicita.

## Riferimenti di accettazione

[Fixture e riproduzione](../fixtures/models/stories15m-q8-v1/README.md). Tre prompt di forward, 32000 logits ciascuno, 15 vettori intermedi dell'ultimo token per prompt e 12 token greedy. Sedici casi tokenizer includono empty, inglese, italiano, CJK, emoji, whitespace, special letterali e NUL.

Reference primaria: calcolo NumPy FP64 sugli esatti pesi Q8_0 ricostruiti. Controprova: runtime ufficiale sui medesimi pesi esportati F32, cache F32, CPU singolo thread, flash attention esclusa. Il file F32 derivato è soltanto un oracle offline; Satori carica il Q8_0 originale.

Gate del futuro forward FP32: valori finiti, errore per elemento ≤`5e-4 + 5e-5*abs(reference)`, RMS per tensore ≤`1e-4`, token greedy identici sulle fixture. T20 usa anche reference locali FP64 degli operatori; un confronto finale da solo non sostituisce le verifiche critiche degli operatori. Le soglie non si allargano per far coincidere il percorso ufficiale con attivazioni quantizzate.

Le differenze tra gli oracoli e gli hash sono nel [manifest](../fixtures/models/stories15m-q8-v1/manifest.json). Il percorso ufficiale Q8_0 è conservato come confronto separato, non come riferimento di aritmetica FP32 con attivazioni non quantizzate.

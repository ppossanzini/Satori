# Tokenizer Stories15M

`StoriesTokenizer` carica e possiede il vocabolario GGUF del [profilo fissato](stories15m-profile-v1.md). Resta utilizzabile dopo la chiusura del file. Hash degli array, tipi, dimensioni, ID speciali e opzioni devono corrispondere al profilo. Un vocabolario modificato richiede un nuovo profilo.

| Operazione | Risultato |
|---|---|
| `Encode(text, addSpecial)` | ID token; BOS opzionale, nessun EOS automatico |
| `Piece(token)` | Span read-only dei byte del pezzo; control e unknown soppressi |
| `DecodeBytes(tokens)` | Byte concatenati, con gestione del prefisso conforme alla decodifica ufficiale |
| `Decode(tokens)` | Testo UTF-8, sequenze invalide sostituite con U+FFFD |
| `CreateDecoder(previousToken)` | Decoder UTF-8 per una continuazione |
| `Append(token)` | Testo disponibile; conserva i byte incompleti tra chiamate |
| `Flush()` | Completa lo stream; un byte UTF-8 incompleto produce U+FFFD |

Il decoder di continuazione rimuove uno spazio iniziale soltanto quando il token precedente è BOS. Non ristampa il prompt. Il controllo dell'arresto EOS appartiene alla generazione; il decoder ne sopprime il pezzo. Nessun chat template è previsto dal profilo.

L'encoding usa una catena di simboli Unicode e una coda di coppie adiacenti. Priorità: score FP32 decrescente, poi posizione sinistra crescente. Le coppie obsolete vengono ignorate. Ogni merge ammesso è già una voce del vocabolario; i simboli residui senza voce diventano byte fallback. Nessuna normalizzazione Unicode aggiuntiva.

## Verifica e provenienza

Quattro test critici coprono i sedici casi ufficiali, identità dei byte di ogni pezzo, decodifica completa, streaming multibyte, limiti input, surrogate isolate, ID invalidi, vocabolario modificato e lifetime. Il test memoria misura tutte le allocazioni della costruzione dopo warm-up: devono restare entro 16 MiB, limite conservativo rispetto alla memoria posseduta.

`fixtures/models/stories15m-q8-v1/tokenizer.gguf` contiene gli esatti metadata tokenizer dell'artefatto fissato e un tensore F32 nullo di un elemento. Non contiene pesi del modello. I test sono eseguibili offline. `tools/tokenizer_fixture.py` rigenera il file dal corpus GGUF verificandone prima SHA-256. Provenienza e hash sono in `tokenizer-manifest.json`, separato dal manifest degli oracoli completi.

Algoritmo verificato contro [llama-vocab.cpp alla revisione fissata](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/src/llama-vocab.cpp). Il runtime Satori non dipende da llama.cpp o SentencePiece. Forward e generazione restano task successivi.

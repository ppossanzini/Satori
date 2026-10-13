# Q4_K_M e UD-IQ

Verifica: 13 ottobre 2026. Raccomandazione: Q4_K_M come primo ampliamento non-ternary; UD-IQ3_XXS come esperimento successivo. UD-IQ2_M/XXS sono candidati di capacità. UD-IQ1 resta esplorativo. Nessun formato aggiuntivo è già eseguibile in Satori.

## Formato e ricetta

Q4_K è un tipo tensoriale GGUF: 256 pesi per blocco, 144 byte, 4,5 bit/peso inclusi scale e minimi. Q4_K_M è una ricetta mista del quantizzatore. Può assegnare Q6_K a tensori sensibili; il costo medio del modello non coincide con quello del blocco Q4_K. Servono anche i tipi floating-point e gli altri tipi presenti nel file concreto.

UD significa Unsloth Dynamic. UD-IQ non è un singolo codec né un nuovo tipo GGUF. La ricetta seleziona precisioni differenti per modello e tensore. IQ2_M è a sua volta una miscela, non un tipo tensoriale omonimo. Il loader deve dispatchare sul tipo del tensore; il prefisso UD serve alla selezione dell'artefatto.

La importance matrix serve alla calibrazione durante la quantizzazione. L'inferenza legge i pesi già quantizzati; non richiede il dataset di calibrazione. IQ usa ricostruzioni non uniformi e, secondo il tipo, codebook, indici e segni. Non si implementa riducendo semplicemente i bit di Q4_0. IQ1/IQ2 non equivalgono al ternary nativo BitNet.

Fonti: [layout ggml](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h), [ricette llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp), [decoder](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c), [documentazione Unsloth](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs.md). I sorgenti llama.cpp sono consultati su master: fissare una revisione prima di portare codice. La pagina Unsloth v2 reindirizza ora a v3 e conserva una sezione v2; il nome UD da solo non identifica la versione della ricetta.

## Artefatti verificati

GB decimali del file, non picco RAM. Revisioni e metadati LFS in [evidenze](quantization-evidence-2026-10-13.json). File Unsloth derivati dai modelli Qwen, distinti dagli artefatti pubblicati da Qwen. Nessun peso scaricato; inventario binario dei tensori ancora da verificare.

| Modello | Q4_K_M | UD-IQ3_XXS | UD-IQ2_M | UD-IQ2_XXS | UD-IQ1_M / S |
|---|---:|---:|---:|---:|---:|
| [Qwen3 4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/22c9fc8a8c7700b76a1789366280a6a5a1ad1120) | 2,497 | 1,674 | 1,533 | 1,255 | 1,143 / 1,083 |
| [Qwen3 8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/a6adef130ffb23ddaf1a62fec9dced968c9bc482) | 5,028 | 3,410 | 3,111 | 2,605 | 2,396 / 2,275 |
| [Qwen2.5-Coder 14B](https://huggingface.co/unsloth/Qwen2.5-Coder-14B-Instruct-GGUF/tree/388f3f20271ef903bb2dbe7d8236158903e7edb3) | 8,988 | — | — | — | — |

Il repository Coder 14B verificato non contiene UD-IQ: non attribuire la disponibilità di Qwen3 a tutta la famiglia. I candidati coding Q4_K_M ufficiali e la KV sono nel [catalogo](dossier.md#candidate-models).

Per Qwen3 8B, UD-IQ3_XXS riduce il file del 32,2% rispetto a Q4_K_M; UD-IQ2_XXS del 48,2%. La KV resta uguale a parità di modello, contesto e precisione. Entrambi i modelli Qwen3 della tabella hanno già pesi Q4_K_M compatibili come dimensione con il budget 16 GB; IQ non è indispensabile alla sola capacità. Il budget totale resta da misurare.

## Costo e rischio per Satori

| Famiglia | Lavoro necessario | Rischio dominante | Priorità proposta |
|---|---|---|---|
| Q4_K_M | Q4_K e Q6_K, inventario dei tipi residui; decoder, dot CPU e Vulkan | Correzioni minimi/scale e miscela tensoriale | Prima estensione: coding 7B, Qwen3 4B, SmolLM2 |
| UD-IQ3_XXS | Codec IQ3_XXS e tutti i tipi della ricetta concreta | Lookup/unpacking e qualità coding | Primo confronto IQ: Qwen3 8B contro lo stesso modello Q4_K_M |
| UD-IQ2_M / XXS | Codec IQ2 pertinenti e tipi misti | Maggiore perdita di qualità; decode non necessariamente più veloce | Esperimento quando il risparmio memoria serve |
| UD-IQ1_M / S | Codec dedicati e miscela completa | Loop, errori di codice/tool calling, qualità instabile | Ricerca separata, nessuna baseline coding |

Il vantaggio di banda può essere annullato da unpacking, lookup e accessi alle tabelle. È un'ipotesi prestazionale da misurare su AVX2 e Vulkan separatamente. Non espandere stabilmente tutti i pesi in F32 per semplificare il supporto.

Le affermazioni Unsloth sulla qualità sono risultati del produttore su modelli e benchmark specifici. Non provano utilità coding sui candidati della tabella. Il ternary nativo resta un asse distinto del requisito R-QUANT.

## Prova minima proposta

1. Leggere header e descrittori tensoriali di un Q4_K_M e un UD-IQ3_XXS, registrando tipo, forma, byte e revisione. Rifiutare tipi mancanti prima della generazione.
2. Verificare decoder e dot su blocchi di riferimento indipendenti, scale/minimi, codebook/segni e bordi. Confrontare scalar, AVX2 e Vulkan sullo stesso calcolo.
3. Confrontare lo stesso modello e tokenizer in Q8_0, Q4_K_M e UD-IQ3_XXS. Valutare compilazione/esecuzione C#, correzione bug, italiano e output lunghi su prompt non usati nella calibrazione.
4. Misurare startup, prefill, TTFT, decode, picco RAM e scratch con contesti 2K/4K. Tenere distinti correttezza numerica e qualità del modello.
5. Aggiungere IQ2 solo se guadagna capacità utile senza scendere sotto i criteri coding approvati. Soglie di qualità e latenza ancora da concordare.

Decisione scouting: estendere Satori usando GGUF/ggml come riferimento di formato; nessuna dipendenza PyTorch o servizio remoto. Il lavoro resta necessario perché gli asset locali coprono Q4_0/Q8_0 e formati ternary custom, non queste ricette.

# Callback stringhe e JIT .NET

Il parser usa GgufStringResult per restituire stringhe da una lettura mappata. Il contenitore sostituisce l'array di una stringa e conserva il contratto della lettura protetta. Il workaround è circoscritto al percorso di caricamento dei metadata.

Ambiente osservato: Arch Linux x64, .NET 10.0.12, build Release. Due esecuzioni della suite sono terminate con SIGSEGV nel thread del compilatore tiered, con AVX2 attivo e disabilitato. Il dump e i simboli corrispondenti ai binari installati individuano un MethodTable nullo in CEEInfo::isExactType, chiamato da Compiler::gtCanSkipCovariantStoreCheck durante l'inlining. Il callback delle stringhe del parser restituisce il valore attraverso un array covariante nel percorso interessato.

Una verifica diagnostica con PGO disabilitato passa. La correzione locale elimina quell'array temporaneo. Le verifiche finali mantengono PGO e tiered compilation predefiniti: 109 test passati con AVX2, 109 senza AVX2 e un ulteriore run standard. Il test di regressione ripete 1000 aperture e letture, verificando nomi e tipi scalari. Non viene modificata la configurazione del runtime o lo SDK del progetto.

La [documentazione Microsoft](https://learn.microsoft.com/en-us/dotnet/core/runtime-config/compilation) definisce le opzioni usate soltanto nella diagnosi. I simboli sono stati scaricati per i build ID esatti dei binari installati. Dump, simboli e strumenti diagnostici temporanei vengono rimossi; nessun nuovo pacchetto runtime entra nella soluzione. Il workaround non implica che ogni causa di crash JIT sia risolta per altri carichi o versioni.

# Pesi memory-mapped

`MappedWeights` apre un file esistente in sola lettura e mappa l'intera lunghezza nel processo a 64 bit. Non carica l'intero contenuto in un array gestito. Un file vuoto viene rifiutato perché non costituisce una regione mappabile. Errori di apertura o mapping rimangono errori I/O; un costruttore fallito rilascia le risorse già acquisite.

## Accesso

`Read<TState>(offset, length, action, state)` espone una `ReadOnlySpan<byte>` soltanto durante la callback. Gli offset sono a 64 bit; ogni finestra ha una lunghezza `int`, come richiesto dagli span. File superiori a 2 GiB si leggono per finestre, senza troncare gli offset. Gli intervalli vengono controllati prima di acquisire il puntatore; un intervallo vuoto alla fine del file è valido.

Lo stato generico supporta anche ref struct per passare dati di lavoro senza closure o boxing. Una callback statica riutilizzata non produce allocazioni gestite nel percorso di accesso dopo warm-up. Il contenuto non viene copiato dal proprietario; una callback può scegliere di copiarlo.

## Chiusura concorrente

Un lock breve rende atomici il controllo dello stato e l'acquisizione del puntatore tramite `SafeMemoryMappedViewHandle.AcquirePointer`. La callback viene eseguita fuori dal lock: più lettori possono procedere contemporaneamente. Il puntatore viene rilasciato in `finally`, anche se la callback solleva un'eccezione.

`Dispose` impedisce immediatamente nuovi accessi e chiude il proprietario. I riferimenti acquisiti dai lettori tengono il mapping valido fino alla fine delle rispettive callback; lo smontaggio effettivo può quindi avvenire dopo il ritorno da `Dispose`. La callback può chiudere il proprietario senza invalidare i propri byte. La chiusura è idempotente.

Nessun puntatore pubblico, span restituito o lease che il chiamante debba rilasciare separatamente. Le operazioni sui byte devono terminare entro la callback sincrona. Le API unsafe del chiamante possono aggirare questo vincolo e non rientrano nel contratto.

## Limiti

Il file deve restare immutato durante la vita del mapping e degli accessi attivi. `FileShare.Read` impedisce aperture incompatibili dove il sistema applica tali vincoli; non garantisce protezione da modifiche o troncamenti esterni su tutti i sistemi. Il mapping non garantisce residenza RAM: il sistema operativo gestisce page cache e I/O.

Questo componente non conosce il formato dei pesi, non valida tensori e non trasferisce dati verso la GPU. `MappedTernary2Matrix` aggiunge il caricamento STQ2 e l'esecuzione tramite gli stessi kernel CPU delle matrici in memoria; il contratto è in [ternary2-v1.md](ternary2-v1.md). Validazione attuale effettuata su Linux x64.

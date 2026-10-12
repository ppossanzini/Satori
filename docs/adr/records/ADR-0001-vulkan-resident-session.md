Id: ADR-0001
Title: Sessione Stories15M residente su Vulkan
Date: 2026-10-12
Status at creation: accepted
Domain: vulkan-resident
Context: Le 43 submission sincrone per token dominano la latenza della baseline.
Decision:
- Pesi Q8 compressi condivisi; KV e scratch indipendenti per sessione.
- Una submission sincrona per append, embedding CPU e readback dei soli logits.
- Riduzioni workgroup da 64 invocazioni, senza dipendenza dai subgroup.
- Operatori FP32 con gate numerici invariati e tabella RoPE calcolata sul CPU.
- Commit dopo fence e readback; errore invalida lo stato, senza replay automatico.
Consequences: Servono operatori registrabili, barriere esplicite e una factory di sessione nel confine Formats. Il budget e l'ABI sono fissati nella vista corrente.
Rejected alternatives: FP64 obbligatorio esclude hardware compatibile; trasferimenti intermedi conservano le attese; fallback a metà token perde il contratto di commit.

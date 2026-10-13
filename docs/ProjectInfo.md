# Satori project documentation

[Product requirements and execution status](../TASKS.md) govern the multi-model roadmap, supported capabilities, quantized formats and representative performance workloads. No regression model defines the product architecture or target.

Reusable technical contracts: [GGUF](gguf-subset-v1.md), [mapped weights](mapped-weights.md), [Vulkan](vulkan-backend-v1.md), [Q8 MatVec](vulkan-q8-matvec.md), [FP32 operators](vulkan-transformer-operators.md).

Model-specific guides and optimization narratives have been removed. Raw historical measurements and regression fixtures remain available as verification data.

[Scouting candidati locali e kernel](scouting/dossier.md): catalogo proposto, budget e audit DS4/AirLLM; nessun nuovo modello è fissato come target del motore.

[Idee applicabili a Satori](scouting/applicability-summary.md): memoria, cache/streaming, kernel e requisiti APEX/Bonsai.

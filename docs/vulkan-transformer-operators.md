# Vulkan FP32 transformer primitives

The shared pipeline records operations into a device-local FP32 arena. Immutable descriptor sets bind the arena and immutable normalization/rotation tables. Every recording method adds a compute read/write barrier and never submits or reads. The synchronous standalone facade provides individual operators; model composition uses the same internal recording path.

RMSNorm uses a 64-lane tree reduction and epsilon 1e-5. One critical hardware test passes independent Float64 fixture values, zero input, profile-size vectors at magnitudes 1e-8/1/100, and feeding the output to cooperative Q8 with its FP64 bound. Operator tolerance stays 2e-6+2e-6*abs(reference). Arbitrary FP32 extremes beyond the profile are not promised. SyncVal: zero errors; Release build: zero warnings/errors.

Context owns the lifetime of registered graphs. Disposing a context releases descriptors, arena and mapped staging before the device. Explicit graph disposal removes its registration; repeated disposal is safe. Shared model tables are introduced at layer composition. [Resident contract](adr/current/vulkan-resident.md).

RoPE uses a host-double sin/cos table rounded to FP32, with disjoint in-place pairs and head dimension 48. Independent fixture position 37 and all 288 channels at positions 0/1/31/63/127 pass the unchanged operator tolerance; two operator tests pass with zero SyncVal errors. No components outside the uploaded/readback vector are exposed.

SwiGLU uses exp(-abs(x)) for a sign-stable sigmoid, with exact gate alias supported. Independent fixture, zero and ±1000 gates, and the first real feed-forward block (GPU RMSNorm, gate/up projections, SwiGLU and down projection) pass unchanged tolerances. Three operator tests pass with zero SyncVal errors. Recording retains gate/up on device; standalone verification transfers are excluded from productive graph composition.

Residual addition is elementwise FP32 with exact input/output aliases and no partial overlap. Distinct output, either exact alias, one-element boundary and normalization chaining pass. Four operator tests pass with zero SyncVal errors; build has no warnings/errors.

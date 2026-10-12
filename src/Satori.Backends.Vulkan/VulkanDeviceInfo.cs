namespace Satori.Backends.Vulkan;

public sealed record VulkanDeviceInfo(int Index, string Name, string Uuid, uint VendorId, uint DeviceId,
    uint ApiVersion, uint DriverVersion, bool Software, uint QueueFamily, uint TimestampValidBits,
    float TimestampPeriod, uint MaxStorageBufferRange, uint MaxDispatchGroups, uint MaxWorkgroupInvocations,
    uint MaxPushConstantsSize, uint MaxStorageDescriptors, uint MaxMemoryAllocations, ulong MaxAllocationBytes, ulong StorageOffsetAlignment, ulong NonCoherentAtomSize,
    IReadOnlyList<uint> MemoryTypeFlags);

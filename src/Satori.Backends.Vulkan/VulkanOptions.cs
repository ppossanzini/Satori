namespace Satori.Backends.Vulkan;

public sealed record VulkanOptions(int? DeviceIndex = null, bool Validation = false);

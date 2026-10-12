namespace Satori.Backends.Vulkan;

public sealed record VulkanAvailability(VulkanAvailabilityReason Reason, string Operation, int? NativeResult = null, IReadOnlyList<string>? Diagnostics = null);

namespace Satori.Backends.Vulkan;

public readonly record struct VulkanMatVecTiming(double HostMilliseconds, double? KernelMilliseconds);

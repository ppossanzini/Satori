namespace Satori.Backends.Vulkan;

public sealed record VulkanQ8Options(bool MeasureKernelTime = false, bool MeasureHostPhases = false, VulkanQ8Kernel Kernel = VulkanQ8Kernel.Row64);

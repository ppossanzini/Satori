namespace Satori.Backends.Vulkan;

/// <summary>Live resident allocations and cumulative successful dispatch measurements.</summary>
public sealed record VulkanOffloadStatistics(int ResidentMatrices, ulong AllocatedBytes, ulong WeightAllocationBytes,
    long CompletedDispatches, long TransferBytes, double HostMilliseconds, double? KernelMilliseconds);

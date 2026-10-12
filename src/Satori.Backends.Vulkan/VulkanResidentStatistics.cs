namespace Satori.Backends.Vulkan;

public sealed record VulkanResidentStatistics(int Matrices,ulong WeightBytes,ulong TableBytes,int Sessions,ulong SessionAllocationBytes,
    long CompletedAppends,long SubmissionCount,long FenceWaitCount,long TransferBytes,long MatVecDispatches);

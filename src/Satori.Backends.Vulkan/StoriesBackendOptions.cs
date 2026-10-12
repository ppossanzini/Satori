namespace Satori.Backends.Vulkan;

/// <summary>Explicit initialization policy; execution failures always propagate.</summary>
public sealed record StoriesBackendOptions(StoriesBackendMode Mode = StoriesBackendMode.Cpu, VulkanOptions? Vulkan = null, StoriesPlannerOptions? Planner = null, VulkanQ8Options? MatVec = null, StoriesVulkanPath Path = StoriesVulkanPath.MatVec);

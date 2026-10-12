using Silk.NET.Vulkan;

namespace Satori.Backends.Vulkan;

public sealed class VulkanException(string operation, Result result) : Exception($"{operation}: {result}")
{
    public string Operation { get; } = operation;
    public Result Result { get; } = result;
}

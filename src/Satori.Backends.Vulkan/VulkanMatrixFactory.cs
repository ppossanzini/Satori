using Satori.Core;

namespace Satori.Backends.Vulkan;

internal sealed class VulkanMatrixFactory(VulkanContext context,VulkanQ8Options options) : IQuantizedMatrixFactory
{
    public IQuantizedMatrix Create(ReadOnlySpan<byte> payload,QuantizedMatrixShape shape) => context.CreateQ8Matrix(payload,shape,options);
}

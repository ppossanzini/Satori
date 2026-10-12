using Satori.Core;

namespace Satori.Backends.Vulkan;

internal sealed class VulkanResidentMatrix(VulkanBuffer weights,QuantizedMatrixShape shape,VulkanQ8Pipeline pipeline)
{
    internal VulkanBuffer Weights {get;}=weights;
    internal QuantizedMatrixShape Shape {get;}=shape;
    internal VulkanQ8Pipeline Pipeline {get;}=pipeline;
}

using Silk.NET.Vulkan;
using Satori.Formats.Gguf;

namespace Satori.Backends.Vulkan;

/// <summary>Optional layer-boundary copies for diagnostic forward calls.</summary>
internal sealed class VulkanResidentCapture : IDisposable
{
    private readonly VulkanFloatGraph _graph;
    private readonly VulkanBuffer _buffer;
    private readonly float[] _host=new float[15*288];
    private readonly List<(string Name,int Offset,int Length)> _records=new(15);
    private int _count;
    internal VulkanResidentCapture(VulkanFloatGraph graph)
    {
        _graph=graph;_buffer=new(graph.Context,15*288*4,BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.HostVisibleBit);
    }
    internal unsafe void Record(string name,uint source,uint length)
    {
        var context=_graph.Context;
        context.Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderWriteBit,PipelineStageFlags.TransferBit,AccessFlags.TransferReadBit);
        var copy=new BufferCopy {SrcOffset=source*4UL,DstOffset=(ulong)_count*4,Size=length*4UL};
        context.Api.CmdCopyBuffer(context.Command,_graph.Arena.Handle,_buffer.Handle,1,in copy);
        context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferReadBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit);
        _records.Add((name,_count,(int)length));_count+=(int)length;
    }
    internal void Complete(StoriesStateObserver observer)
    {
        _buffer.Read(_host.AsSpan(0,_count));
        foreach(var record in _records)observer(record.Name,_host.AsSpan(record.Offset,record.Length));
    }
    public void Dispose()=>_buffer.Dispose();
}

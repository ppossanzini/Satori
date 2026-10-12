using System.Diagnostics;
using Silk.NET.Vulkan;
using Satori.Core;

namespace Satori.Backends.Vulkan;

/// <summary>Resident immutable Q8_0 weights, with owned vector scratch borrowing a context's lifetime.</summary>
public sealed unsafe class VulkanQ8Matrix : IQuantizedMatrix
{
    private readonly VulkanContext _context;
    private readonly VulkanQ8Pipeline _pipeline;
    private readonly List<VulkanBuffer> _buffers=new(5);
    private DescriptorPool _pool;
    private DescriptorSet _set;
    private QueryPool _queries;
    private bool _disposed;
    private readonly bool _measureHostPhases;
    internal VulkanHostPhases HostPhases { get; private set; }
    internal long PhaseSamples { get; private set; }
    internal long CompletedDispatches { get; private set; }
    internal long TransferBytes { get; private set; }
    internal double HostMilliseconds { get; private set; }
    internal long KernelSamples { get; private set; }
    internal double KernelMilliseconds { get; private set; }
    void IQuantizedMatrix.Execute(ReadOnlySpan<float> input, Span<float> output) => Execute(input,output);
    public QuantizedMatrixShape Shape { get; }
    public ulong WeightAllocationBytes { get { lock(_context.Sync) return _disposed?0:_buffers[0].AllocationBytes; } }
    public ulong AllocatedBytes { get { lock(_context.Sync) return _buffers.Aggregate(0UL,(sum,b)=>sum+b.AllocationBytes); } }

    internal VulkanQ8Matrix(VulkanContext context, VulkanQ8Pipeline pipeline, ReadOnlySpan<byte> payload, QuantizedMatrixShape shape, VulkanQ8Options options)
    {
        _context=context; _pipeline=pipeline; Shape=shape; _measureHostPhases=options.MeasureHostPhases;
        try
        {
            int padded=checked((payload.Length+3)&~3);
            _buffers.Add(new(context,(ulong)padded,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.DeviceLocalBit));
            using(var staging=new VulkanBuffer(context,(ulong)padded,BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.HostVisibleBit))
            {
                staging.WriteBytes(payload,padded-payload.Length);
                context.BeginCommands();
                var copy=new BufferCopy {Size=(ulong)padded};
                context.Api.CmdCopyBuffer(context.Command,staging.Handle,_buffers[0].Handle,1,in copy);
                context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit);
                context.SubmitAndWait();
            }
            _buffers.Add(new(context,(ulong)shape.Columns*4,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.DeviceLocalBit));
            _buffers.Add(new(context,(ulong)shape.Rows*4,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.DeviceLocalBit));
            _buffers.Add(new(context,(ulong)shape.Columns*4,BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.HostVisibleBit));
            _buffers.Add(new(context,(ulong)shape.Rows*4,BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.HostVisibleBit));
            var size=new DescriptorPoolSize {Type=DescriptorType.StorageBuffer,DescriptorCount=3};
            var createPool=new DescriptorPoolCreateInfo {SType=StructureType.DescriptorPoolCreateInfo,MaxSets=1,PoolSizeCount=1,PPoolSizes=&size};
            context.Check(context.Api.CreateDescriptorPool(context.Device,in createPool,null,out _pool),"Create Q8 descriptor pool");
            var layout=pipeline.SetLayout;
            var allocate=new DescriptorSetAllocateInfo {SType=StructureType.DescriptorSetAllocateInfo,DescriptorPool=_pool,DescriptorSetCount=1,PSetLayouts=&layout};
            context.Check(context.Api.AllocateDescriptorSets(context.Device,in allocate,out _set),"Allocate Q8 descriptor set");
            var infos=stackalloc DescriptorBufferInfo[3];
            var writes=stackalloc WriteDescriptorSet[3];
            ulong* lengths=stackalloc ulong[3] {(ulong)padded,(ulong)shape.Columns*4,(ulong)shape.Rows*4};
            for(uint i=0;i<3;i++)
            {
                infos[i]=new() {Buffer=_buffers[(int)i].Handle,Offset=0,Range=lengths[i]};
                writes[i]=new() {SType=StructureType.WriteDescriptorSet,DstSet=_set,DstBinding=i,DescriptorCount=1,DescriptorType=DescriptorType.StorageBuffer,PBufferInfo=&infos[i]};
            }
            context.Api.UpdateDescriptorSets(context.Device,3,writes,0,null);
            if(options.MeasureKernelTime && context.Info.TimestampValidBits>0)
            {
                var query=new QueryPoolCreateInfo {SType=StructureType.QueryPoolCreateInfo,QueryType=QueryType.Timestamp,QueryCount=2};
                context.Check(context.Api.CreateQueryPool(context.Device,in query,null,out _queries),"Create Q8 timestamp pool");
            }
            context.ThrowIfValidationFailed();
        }
        catch { DisposeCore(); throw; }
    }

    /// <summary>Uploads only the vector and reads the result; timestamps are optional and exclude host transfers.</summary>
    public VulkanMatVecTiming Execute(ReadOnlySpan<float> input, Span<float> output)
    {
        lock(_context.Sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            _context.EnsureReady();
            if(input.Length!=Shape.Columns || output.Length!=Shape.Rows || input.Overlaps(output)) throw new ArgumentException("Matrix/vector dimensions or aliasing invalid.");
            foreach(float value in input) if(!float.IsFinite(value)) throw new ArgumentException("Expected finite FP32 input.",nameof(input));
            long start=Stopwatch.GetTimestamp();
            _buffers[3].Write(input);
            long uploaded=_measureHostPhases?Stopwatch.GetTimestamp():0;
            _context.BeginCommands();
            if(_queries.Handle!=0) _context.Api.CmdResetQueryPool(_context.Command,_queries,0,2);
            var copy=new BufferCopy {Size=(ulong)Shape.Columns*4};
            _context.Api.CmdCopyBuffer(_context.Command,_buffers[3].Handle,_buffers[1].Handle,1,in copy);
            _context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit);
            _context.Api.CmdBindPipeline(_context.Command,PipelineBindPoint.Compute,_pipeline.Handle);
            var set=_set;
            _context.Api.CmdBindDescriptorSets(_context.Command,PipelineBindPoint.Compute,_pipeline.Layout,0,1,in set,0,null);
            uint* dimensions=stackalloc uint[2] {(uint)Shape.Rows,(uint)Shape.Columns};
            _context.Api.CmdPushConstants(_context.Command,_pipeline.Layout,ShaderStageFlags.ComputeBit,0,8,dimensions);
            if(_queries.Handle!=0) _context.Api.CmdWriteTimestamp(_context.Command,PipelineStageFlags.ComputeShaderBit,_queries,0);
            _context.Api.CmdDispatch(_context.Command,((uint)Shape.Rows+63)/64,1,1);
            if(_queries.Handle!=0) _context.Api.CmdWriteTimestamp(_context.Command,PipelineStageFlags.ComputeShaderBit,_queries,1);
            _context.Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderWriteBit,PipelineStageFlags.TransferBit,AccessFlags.TransferReadBit);
            copy.Size=(ulong)Shape.Rows*4;
            _context.Api.CmdCopyBuffer(_context.Command,_buffers[2].Handle,_buffers[4].Handle,1,in copy);
            _context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.HostBit,AccessFlags.HostReadBit);
            long recorded=_measureHostPhases?Stopwatch.GetTimestamp():0;
            var submission=_context.SubmitAndWait(_measureHostPhases);
            long waited=_measureHostPhases?Stopwatch.GetTimestamp():0;
            double? kernel=ReadKernelTime();
            long queried=_measureHostPhases?Stopwatch.GetTimestamp():0;
            _buffers[4].Read(output);
            long end=Stopwatch.GetTimestamp();
            double host=Stopwatch.GetElapsedTime(start,end).TotalMilliseconds;
            if(_measureHostPhases)
            {
                double upload=Stopwatch.GetElapsedTime(start,uploaded).TotalMilliseconds;
                double recording=Stopwatch.GetElapsedTime(uploaded,recorded).TotalMilliseconds;
                double query=Stopwatch.GetElapsedTime(waited,queried).TotalMilliseconds;
                double readback=Stopwatch.GetElapsedTime(queried,end).TotalMilliseconds;
                double other=host-upload-recording-query-readback-submission.PreparationMs-submission.SubmitMs-submission.WaitMs-submission.ValidationMs;
                HostPhases+=new VulkanHostPhases(upload,recording,submission.PreparationMs,submission.SubmitMs,submission.WaitMs,submission.ValidationMs,query,readback,other);
                PhaseSamples++;
            }
            CompletedDispatches++;
            TransferBytes+=((long)Shape.Columns+Shape.Rows)*4;
            HostMilliseconds+=host;
            KernelMilliseconds+=kernel??0;
            if(kernel.HasValue) KernelSamples++;
            return new(host,kernel);
        }
    }

    private double? ReadKernelTime()
    {
        if(_queries.Handle==0) return null;
        ulong* values=stackalloc ulong[4];
        var result=_context.Api.GetQueryPoolResults(_context.Device,_queries,0,2,32,values,16,QueryResultFlags.Result64Bit|QueryResultFlags.ResultWithAvailabilityBit);
        if(result==Result.NotReady) return null;
        _context.Check(result,"Read Q8 timestamp results");
        if(values[1]==0 || values[3]==0) return null;
        uint bits=_context.Info.TimestampValidBits;
        ulong mask=bits==64?ulong.MaxValue:(1UL<<(int)bits)-1;
        ulong ticks=unchecked(values[2]-values[0])&mask;
        return ticks*(double)_context.Info.TimestampPeriod/1_000_000;
    }

    public void Dispose()
    {
        lock(_context.Sync) { DisposeCore(); _context.Forget(this); }
    }

    internal void DisposeCore()
    {
        if(_disposed) return;
        _context.CompletePendingForDisposal();
        _disposed=true;
        if(_queries.Handle!=0) { _context.Api.DestroyQueryPool(_context.Device,_queries,null); _queries=default; }
        if(_pool.Handle!=0) { _context.Api.DestroyDescriptorPool(_context.Device,_pool,null); _pool=default; }
        foreach(var buffer in _buffers) buffer.Dispose();
        _buffers.Clear();
    }
}

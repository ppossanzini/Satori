using Silk.NET.Vulkan;

namespace Satori.Backends.Vulkan;

/// <summary>Owns FP32 arena/staging and immutable descriptors for synchronous operator recording.</summary>
internal sealed unsafe class VulkanFloatGraph : IDisposable
{
    internal readonly VulkanContext Context;
    internal VulkanBuffer Arena = null!, Upload = null!, Readback = null!, Tables = null!;
    internal VulkanFloatPipeline Pipeline = null!;
    private DescriptorPool _pool;
    private DescriptorSet _set;
    private bool _disposed, _ownsTables;
    internal ulong AllocatedBytes => Arena.AllocationBytes+Upload.AllocationBytes+Readback.AllocationBytes+(_ownsTables?Tables.AllocationBytes:0);

    internal VulkanFloatGraph(VulkanContext context,int words,ReadOnlySpan<float> tables,int uploadWords,int readbackWords)
    {
        Context=context; _ownsTables=true;
        lock(context.Sync)
        {
            context.EnsureReady();
            try
            {
                Pipeline=new(context);
                Tables=new(context,(ulong)tables.Length*4,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.DeviceLocalBit);
                using(var staging=new VulkanBuffer(context,(ulong)tables.Length*4,BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.HostVisibleBit))
                {
                    staging.Write(tables);context.BeginCommands();
                    var copy=new BufferCopy {Size=(ulong)tables.Length*4};
                    context.Api.CmdCopyBuffer(context.Command,staging.Handle,Tables.Handle,1,in copy);
                    context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit);
                    context.SubmitAndWait();
                }
                Initialize(words,uploadWords,readbackWords);
            }
            catch { Dispose();throw; }
        }
    }

    internal VulkanFloatGraph(VulkanContext context,int words,VulkanBuffer tables,VulkanFloatPipeline pipeline,int uploadWords,int readbackWords)
    {
        Context=context;Tables=tables;Pipeline=pipeline;
        lock(context.Sync) {context.EnsureReady();try {Initialize(words,uploadWords,readbackWords);}catch {Dispose();throw;}}
    }
    private void Initialize(int words,int uploadWords,int readbackWords)
    {
        Arena=new(Context,(ulong)words*4,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferSrcBit|BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.DeviceLocalBit);
        Upload=new(Context,(ulong)uploadWords*4,BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.HostVisibleBit);
        Readback=new(Context,(ulong)readbackWords*4,BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.HostVisibleBit);
        var size=new DescriptorPoolSize {Type=DescriptorType.StorageBuffer,DescriptorCount=2};
        var create=new DescriptorPoolCreateInfo {SType=StructureType.DescriptorPoolCreateInfo,MaxSets=1,PoolSizeCount=1,PPoolSizes=&size};
        Context.Check(Context.Api.CreateDescriptorPool(Context.Device,in create,null,out _pool),"Create FP32 descriptors");
        var layout=Pipeline.SetLayout;
        var allocate=new DescriptorSetAllocateInfo {SType=StructureType.DescriptorSetAllocateInfo,DescriptorPool=_pool,DescriptorSetCount=1,PSetLayouts=&layout};
        Context.Check(Context.Api.AllocateDescriptorSets(Context.Device,in allocate,out _set),"Allocate FP32 descriptors");
        var infos=stackalloc DescriptorBufferInfo[2];var writes=stackalloc WriteDescriptorSet[2];
        for(uint i=0;i<2;i++)
        {
            var buffer=i==0?Arena:Tables;
            infos[i]=new() {Buffer=buffer.Handle,Range=Vk.WholeSize};
            writes[i]=new() {SType=StructureType.WriteDescriptorSet,DstSet=_set,DstBinding=i,DescriptorCount=1,DescriptorType=DescriptorType.StorageBuffer,PBufferInfo=&infos[i]};
        }
        Context.Api.UpdateDescriptorSets(Context.Device,2,writes,0,null);
        Context.ThrowIfValidationFailed();Context.Register(this);
    }

    internal void EnsureReady() { ObjectDisposedException.ThrowIf(_disposed,this);Context.EnsureReady(); }
    internal void RecordUpload(uint destination,uint words,uint source=0)
    {
        var copy=new BufferCopy {SrcOffset=source*4UL,DstOffset=destination*4UL,Size=words*4UL};
        Context.Api.CmdCopyBuffer(Context.Command,Upload.Handle,Arena.Handle,1,in copy);
        Context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit);
    }
    internal void Record(uint op,uint a,uint b,uint c,uint length,uint position=0,uint layer=0,uint table=0)
    {
        Context.Api.CmdBindPipeline(Context.Command,PipelineBindPoint.Compute,Pipeline.Handle);
        var set=_set;Context.Api.CmdBindDescriptorSets(Context.Command,PipelineBindPoint.Compute,Pipeline.Layout,0,1,in set,0,null);
        uint* push=stackalloc uint[8] {op,a,b,c,length,position,layer,table};
        Context.Api.CmdPushConstants(Context.Command,Pipeline.Layout,ShaderStageFlags.ComputeBit,0,32,push);
        Context.Api.CmdDispatch(Context.Command,op==4?length/48:1,1,1);
        Context.Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit);
    }
    internal void RecordDownload(uint source,uint words)
    {
        Context.Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderWriteBit,PipelineStageFlags.TransferBit,AccessFlags.TransferReadBit);
        var copy=new BufferCopy {SrcOffset=source*4UL,Size=words*4UL};
        Context.Api.CmdCopyBuffer(Context.Command,Arena.Handle,Readback.Handle,1,in copy);
        Context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.HostBit,AccessFlags.HostReadBit);
    }
    internal void Clear()
    {
        EnsureReady();Context.BeginCommands();Context.Api.CmdFillBuffer(Context.Command,Arena.Handle,0,Vk.WholeSize,0);
        Context.Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit);
        Context.SubmitAndWait();
    }
    public void Dispose()
    {
        lock(Context.Sync)
        {
            if(_disposed)return;_disposed=true;Context.CompletePendingForDisposal();
            if(_pool.Handle!=0) {Context.Api.DestroyDescriptorPool(Context.Device,_pool,null);_pool=default;}
            Arena?.Dispose();Upload?.Dispose();Readback?.Dispose();
            if(_ownsTables) {Tables?.Dispose();Pipeline?.Dispose();}
            Context.Unregister(this);
        }
    }
}

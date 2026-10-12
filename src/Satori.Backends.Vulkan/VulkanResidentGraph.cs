using Silk.NET.Vulkan;

namespace Satori.Backends.Vulkan;

/// <summary>Private KV/activation arena with immutable matrix bindings borrowing shared weights.</summary>
internal sealed unsafe class VulkanResidentGraph : IDisposable
{
    internal const uint X=0,Norm=288,Q=576,Attention=864,Projected=1152,Gate=1440,Up=2208,Logits=2976,Keys=34976,Values=256160,Words=477344;
    private readonly VulkanResidentModel _model;
    internal readonly VulkanFloatGraph Float;
    private DescriptorPool _pool;
    private readonly DescriptorSet[] _sets=new DescriptorSet[43];
    private bool _disposed;
    internal ulong AllocatedBytes=>Float.AllocatedBytes;
    internal VulkanResidentGraph(VulkanResidentModel model)
    {
        _model=model;var context=model.Context;
        Float=new(context,(int)Words,model.Tables,model.FloatPipeline,288,32000);
        try
        {
            var size=new DescriptorPoolSize {Type=DescriptorType.StorageBuffer,DescriptorCount=129};
            var create=new DescriptorPoolCreateInfo {SType=StructureType.DescriptorPoolCreateInfo,MaxSets=43,PoolSizeCount=1,PPoolSizes=&size};
            context.Check(context.Api.CreateDescriptorPool(context.Device,in create,null,out _pool),"Create resident matrix descriptors");
            var infos=stackalloc DescriptorBufferInfo[3];var writes=stackalloc WriteDescriptorSet[3];
            for(int index=0;index<43;index++)
            {
                var matrix=model.Matrices[index];var layout=matrix.Pipeline.SetLayout;
                var allocate=new DescriptorSetAllocateInfo {SType=StructureType.DescriptorSetAllocateInfo,DescriptorPool=_pool,DescriptorSetCount=1,PSetLayouts=&layout};
                context.Check(context.Api.AllocateDescriptorSets(context.Device,in allocate,out _sets[index]),"Allocate resident matrix descriptor");
                for(uint binding=0;binding<3;binding++)
                {
                    infos[binding]=new() {Buffer=binding==0?matrix.Weights.Handle:Float.Arena.Handle,Range=Vk.WholeSize};
                    writes[binding]=new() {SType=StructureType.WriteDescriptorSet,DstSet=_sets[index],DstBinding=binding,DescriptorCount=1,DescriptorType=DescriptorType.StorageBuffer,PBufferInfo=&infos[binding]};
                }
                context.Api.UpdateDescriptorSets(context.Device,3,writes,0,null);
            }
            context.ThrowIfValidationFailed();
        }
        catch {Dispose();throw;}
    }
    internal void Begin(ReadOnlySpan<float> input)
    {
        _model.EnsureReady();Float.EnsureReady();Float.Upload.Write(input);_model.Context.BeginCommands();Float.RecordUpload(X,288);
    }
    internal void MatVec(int index,uint input,uint output)
    {
        var context=_model.Context;var matrix=_model.Matrices[index];var pipeline=matrix.Pipeline;var set=_sets[index];
        context.Api.CmdBindPipeline(context.Command,PipelineBindPoint.Compute,pipeline.Handle);
        context.Api.CmdBindDescriptorSets(context.Command,PipelineBindPoint.Compute,pipeline.Layout,0,1,in set,0,null);
        uint* push=stackalloc uint[4] {(uint)matrix.Shape.Rows,(uint)matrix.Shape.Columns,input,output};
        context.Api.CmdPushConstants(context.Command,pipeline.Layout,ShaderStageFlags.ComputeBit,0,16,push);
        context.Api.CmdDispatch(context.Command,pipeline.Kernel==VulkanQ8Kernel.Cooperative64?(uint)matrix.Shape.Rows:((uint)matrix.Shape.Rows+63)/64,1,1);
        context.Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit|AccessFlags.ShaderWriteBit);
    }
    internal void Layer(int layer,int position)
    {
        int first=layer*7;uint key=Keys+(uint)(layer*128*288),value=Values+(uint)(layer*128*288),current=(uint)(position*288);
        Float.Record(0,X,(uint)(layer*576),Norm,288);
        MatVec(first,Norm,Q);MatVec(first+1,Norm,key+current);MatVec(first+2,Norm,value+current);
        Float.Record(1,Q,0,0,288,(uint)position,table:VulkanResidentModel.RopeTable);
        Float.Record(1,key+current,0,0,288,(uint)position,table:VulkanResidentModel.RopeTable);
        Float.Record(4,Q,key,Attention,288,(uint)(position+1),value);
        MatVec(first+3,Attention,Projected);Float.Record(3,X,Projected,X,288);
        Float.Record(0,X,(uint)(layer*576+288),Norm,288);
        MatVec(first+4,Norm,Gate);MatVec(first+5,Norm,Up);Float.Record(2,Gate,Up,Gate,768);
        MatVec(first+6,Gate,Projected);Float.Record(3,X,Projected,X,288);
    }
    internal void Finish(uint source,Span<float> output,int matvecs)
    {
        Float.RecordDownload(source,(uint)output.Length);_model.Context.SubmitAndWait();Float.Readback.Read(output);_model.Commit(output.Length,matvecs);
    }
    public void Dispose()
    {
        lock(_model.Context.Sync)
        {
            if(_disposed)return;_disposed=true;_model.Context.CompletePendingForDisposal();
            if(_pool.Handle!=0){_model.Context.Api.DestroyDescriptorPool(_model.Context.Device,_pool,null);_pool=default;}
            Float.Dispose();
        }
    }
}

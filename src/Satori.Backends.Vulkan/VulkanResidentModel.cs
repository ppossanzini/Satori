using System.Runtime.InteropServices;
using Satori.Core;
using Satori.Formats.Gguf;

namespace Satori.Backends.Vulkan;

/// <summary>Shared compressed Stories15M weights and immutable FP32 operator tables.</summary>
public sealed class VulkanResidentModel : IDisposable
{
    internal readonly VulkanContext Context;
    internal readonly List<VulkanResidentMatrix> Matrices=[];
    internal VulkanBuffer Tables=null!;
    internal VulkanFloatPipeline FloatPipeline=null!;
    private VulkanQ8Pipeline? _row,_cooperative;
    private readonly HashSet<VulkanResidentLayerSession> _sessions=[];
    private bool _disposed;
    private long _appends,_transfers,_matvecs;
    internal const uint RopeTable=13*288;
    public VulkanResidentModel(string path,VulkanContext context)
    {
        Context=context;
        lock(context.Sync)
        {
            context.EnsureReady();
            try
            {
                using var validated=new StoriesModel(path);
                using var file=new GgufFile(path);
                _row=new(context,VulkanQ8Kernel.Row64,true);_cooperative=new(context,VulkanQ8Kernel.Cooperative64,true);FloatPipeline=new(context);
                float[] norm=new float[13*288];
                for(int layer=0;layer<6;layer++)
                {
                    new GgufFloatVector(file,$"blk.{layer}.attn_norm.weight").Values.CopyTo(norm.AsSpan(layer*576));
                    new GgufFloatVector(file,$"blk.{layer}.ffn_norm.weight").Values.CopyTo(norm.AsSpan(layer*576+288));
                    foreach(string name in new[] {"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"}) Add(file,$"blk.{layer}.{name}.weight");
                }
                new GgufFloatVector(file,"output_norm.weight").Values.CopyTo(norm.AsSpan(12*288));Add(file,"output.weight");
                Tables=VulkanBuffer.UploadDevice(context,MemoryMarshal.AsBytes(VulkanTransformerOperators.CreateTables(norm).AsSpan()));
                context.ThrowIfValidationFailed();context.Register(this);
            }
            catch {Dispose();throw;}
        }
    }
    private void Add(GgufFile file,string name)
    {
        var matrix=new GgufQuantizedMatrix(file,name);
        file.ReadTensor(name,0,matrix.Shape.ByteLength,(bytes,model)=>
        {
            var buffer=VulkanBuffer.UploadDevice(Context,bytes);
            try {model.Matrices.Add(new(buffer,matrix.Shape,matrix.Shape.Rows==32000?_row!:_cooperative!));}
            catch {buffer.Dispose();throw;}
        },this);
    }
    internal void EnsureReady(){ObjectDisposedException.ThrowIf(_disposed,this);Context.EnsureReady();}
    public VulkanResidentLayerSession CreateLayerSession(int layer)
    {
        lock(Context.Sync){EnsureReady();if((uint)layer>=6)throw new ArgumentOutOfRangeException(nameof(layer));var session=new VulkanResidentLayerSession(this,layer);_sessions.Add(session);return session;}
    }
    internal void Forget(VulkanResidentLayerSession session)=>_sessions.Remove(session);
    internal void Commit(int outputWords,int matvecs){_appends++;_transfers+=(288L+outputWords)*4;_matvecs+=matvecs;}
    public VulkanResidentStatistics Statistics
    {
        get {lock(Context.Sync){EnsureReady();return new(Matrices.Count,Matrices.Aggregate(0UL,(sum,m)=>sum+m.Weights.AllocationBytes),Tables.AllocationBytes,
            _sessions.Count,_sessions.Aggregate(0UL,(sum,s)=>sum+s.AllocatedBytes),_appends,_appends,_appends,_transfers,_matvecs);}}
    }
    public void Dispose()
    {
        lock(Context.Sync)
        {
            if(_disposed)return;_disposed=true;Context.CompletePendingForDisposal();
            foreach(var session in _sessions.ToArray())session.Dispose();_sessions.Clear();
            foreach(var matrix in Matrices)matrix.Weights.Dispose();Matrices.Clear();
            Tables?.Dispose();FloatPipeline?.Dispose();_row?.Dispose();_cooperative?.Dispose();Context.Unregister(this);
        }
    }
}

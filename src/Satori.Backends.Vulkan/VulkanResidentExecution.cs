using Satori.Formats.Gguf;

namespace Satori.Backends.Vulkan;

internal sealed class VulkanResidentExecution : IStoriesSessionExecution
{
    private readonly VulkanResidentModel _model;
    private readonly VulkanResidentGraph _graph;
    private int _count;
    private bool _failed,_disposed;
    internal ulong AllocatedBytes=>_graph.AllocatedBytes;
    internal VulkanResidentExecution(VulkanResidentModel model){_model=model;_graph=new(model);}
    public void Append(ReadOnlySpan<float> embedding,int position,Span<float> logits,StoriesStateObserver? observer=null)
    {
        lock(_model.Context.Sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);_model.EnsureReady();
            if(_failed)throw new InvalidOperationException("Resident session failed; reset before reuse.");
            if(position!=_count || position>=128 || embedding.Length!=288 || logits.Length!=32000)throw new ArgumentException("Invalid resident append dimensions or position.");
            try
            {
                using var capture=observer is null?null:new VulkanResidentCapture(_graph.Float);
                _graph.Begin(embedding);for(int layer=0;layer<6;layer++)_graph.Layer(layer,position,capture);
                _graph.Float.Record(0,VulkanResidentGraph.X,12*288,VulkanResidentGraph.Norm,288);
                capture?.Record("result_norm",VulkanResidentGraph.Norm,288);
                _graph.MatVec(42,VulkanResidentGraph.Norm,VulkanResidentGraph.Logits);
                _graph.Finish(VulkanResidentGraph.Logits,logits,43);if(observer is not null)capture!.Complete(observer);_count++;
            }
            catch{_failed=true;_count=0;throw;}
        }
    }
    public void Reset(){lock(_model.Context.Sync){ObjectDisposedException.ThrowIf(_disposed,this);_model.EnsureReady();_graph.Float.Clear();_count=0;_failed=false;}}
    public void Dispose(){lock(_model.Context.Sync){if(_disposed)return;_disposed=true;_graph.Dispose();_model.Forget(this);}}
}

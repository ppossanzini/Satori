namespace Satori.Backends.Vulkan;

/// <summary>One resident layer with private sequence state, borrowing model lifetime.</summary>
public sealed class VulkanResidentLayerSession : IDisposable
{
    private readonly VulkanResidentModel _model;
    private readonly VulkanResidentGraph _graph;
    private readonly int _layer;
    private int _count;
    private bool _disposed,_failed;
    internal ulong AllocatedBytes=>_graph.AllocatedBytes;
    internal VulkanResidentLayerSession(VulkanResidentModel model,int layer){_model=model;_layer=layer;_graph=new(model);}
    public int Count {get {lock(_model.Context.Sync){EnsureReady();return _count;}}}
    private void EnsureReady(){ObjectDisposedException.ThrowIf(_disposed,this);_model.EnsureReady();}
    public bool TryAppend(ReadOnlySpan<float> input,Span<float> output)
    {
        lock(_model.Context.Sync)
        {
            EnsureReady();if(_failed)throw new InvalidOperationException("Layer session failed; reset before reuse.");
            if(input.Length!=288 || output.Length!=288)throw new ArgumentException("Expected 288 channels.");
            if(_count==128)return false;
            try{_graph.Begin(input);_graph.Layer(_layer,_count);_graph.Finish(VulkanResidentGraph.X,output,7);_count++;return true;}
            catch{_count=0;_failed=true;throw;}
        }
    }
    public void Reset(){lock(_model.Context.Sync){EnsureReady();_graph.Float.Clear();_count=0;_failed=false;}}
    public void Dispose(){lock(_model.Context.Sync){if(_disposed)return;_disposed=true;_graph.Dispose();_model.Forget(this);}}
}

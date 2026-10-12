namespace Satori.Backends.Vulkan;

/// <summary>Single-layer causal attention with private FP32 K/V and synchronous append commit.</summary>
public sealed class VulkanAttentionSession : IDisposable
{
    private readonly VulkanFloatGraph _graph;
    private readonly int _width;
    private readonly float[] _host;
    private int _count;
    private bool _failed;
    public int CacheBytes => 2*128*_width*4;
    public int Count {get {lock(_graph.Context.Sync){_graph.EnsureReady();return _count;}}}
    public VulkanAttentionSession(VulkanContext context,int width=288)
    {
        if(width<=0 || width>288 || width%48!=0)throw new ArgumentOutOfRangeException(nameof(width));
        _width=width;_host=new float[3*width];
        _graph=new(context,2*128*width+2*width,[0f],3*width,width);
    }
    public bool TryAppend(ReadOnlySpan<float> query,ReadOnlySpan<float> key,ReadOnlySpan<float> value,Span<float> output)
    {
        lock(_graph.Context.Sync)
        {
            _graph.EnsureReady();
            if(_failed)throw new InvalidOperationException("Attention session failed; reset before reuse.");
            if(query.Length!=_width || key.Length!=_width || value.Length!=_width || output.Length!=_width)throw new ArgumentException("Attention dimensions differ.");
            if(_count==128)return false;
            try
            {
                query.CopyTo(_host);key.CopyTo(_host.AsSpan(_width));value.CopyTo(_host.AsSpan(2*_width));_graph.Upload.Write(_host);
                var context=_graph.Context;context.BeginCommands();
                uint keys=(uint)(2*_width),values=keys+(uint)(128*_width),position=(uint)(_count*_width);
                _graph.RecordUpload(0,(uint)_width);_graph.RecordUpload(keys+position,(uint)_width,(uint)_width);_graph.RecordUpload(values+position,(uint)_width,(uint)(2*_width));
                _graph.Record(4,0,keys,(uint)_width,(uint)_width,(uint)(_count+1),values);
                _graph.RecordDownload((uint)_width,(uint)_width);context.SubmitAndWait();_graph.Readback.Read(output);
                _count++;return true;
            }
            catch {_count=0;_failed=true;throw;}
        }
    }
    public void Reset()
    {
        lock(_graph.Context.Sync){_graph.Clear();_count=0;_failed=false;}
    }
    public void Dispose() => _graph.Dispose();
}

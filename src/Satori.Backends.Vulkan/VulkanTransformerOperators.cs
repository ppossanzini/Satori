namespace Satori.Backends.Vulkan;

/// <summary>Synchronous FP32 transformer primitives sharing an immutable weight/rotation table.</summary>
public sealed class VulkanTransformerOperators : IDisposable
{
    private readonly VulkanFloatGraph _graph;
    private readonly int _width;
    private readonly float[] _host = new float[2304];
    internal static float[] CreateTables(ReadOnlySpan<float> weights)
    {
        float[] table=new float[weights.Length+128*24*2];weights.CopyTo(table);
        for(int pos=0;pos<128;pos++) for(int j=0;j<24;j++)
        {
            var (sin,cos)=Math.SinCos(pos/Math.Pow(10000,2.0*j/48));
            table[weights.Length+(pos*24+j)*2]=(float)sin;table[weights.Length+(pos*24+j)*2+1]=(float)cos;
        }
        return table;
    }
    public VulkanTransformerOperators(VulkanContext context,ReadOnlySpan<float> normalizationWeights)
    {
        if(normalizationWeights.IsEmpty || normalizationWeights.Length>768)throw new ArgumentOutOfRangeException(nameof(normalizationWeights));
        _width=normalizationWeights.Length;
        _graph=new(context,2304,CreateTables(normalizationWeights),2304,768);
    }
    public void RmsNorm(ReadOnlySpan<float> input,Span<float> output)
    {
        lock(_graph.Context.Sync)
        {
            _graph.EnsureReady();
            if(input.Length!=_width || output.Length!=_width)throw new ArgumentException("Normalization dimensions differ.");
            input.CopyTo(_host);_graph.Upload.Write(_host.AsSpan(0,_width));
            _graph.Context.BeginCommands();_graph.RecordUpload(0,(uint)_width);
            _graph.Record(0,0,0,1536,(uint)_width);_graph.RecordDownload(1536,(uint)_width);
            _graph.Context.SubmitAndWait();_graph.Readback.Read(output);
        }
    }
    public void Rope(Span<float> vector,int position)
    {
        lock(_graph.Context.Sync)
        {
            _graph.EnsureReady();
            if(vector.IsEmpty || vector.Length>768 || vector.Length%48!=0 || (uint)position>=128)throw new ArgumentOutOfRangeException(nameof(position));
            vector.CopyTo(_host);_graph.Upload.Write(_host.AsSpan(0,vector.Length));
            _graph.Context.BeginCommands();_graph.RecordUpload(0,(uint)vector.Length);
            _graph.Record(1,0,0,0,(uint)vector.Length,(uint)position,table:(uint)_width);_graph.RecordDownload(0,(uint)vector.Length);
            _graph.Context.SubmitAndWait();_graph.Readback.Read(vector);
        }
    }
    public void SwiGlu(ReadOnlySpan<float> gate,ReadOnlySpan<float> up,Span<float> output) => Binary(2,gate,up,output);
    private void Binary(uint op,ReadOnlySpan<float> left,ReadOnlySpan<float> right,Span<float> output)
    {
        lock(_graph.Context.Sync)
        {
            _graph.EnsureReady();
            if(left.IsEmpty || left.Length>768 || right.Length!=left.Length || output.Length!=left.Length)throw new ArgumentException("Operator dimensions differ.");
            if(left.Overlaps(output,out int lo) && lo!=0 || right.Overlaps(output,out int ro) && ro!=0)throw new ArgumentException("Partial overlap.");
            left.CopyTo(_host);right.CopyTo(_host.AsSpan(768));_graph.Upload.Write(_host.AsSpan(0,768+right.Length));
            _graph.Context.BeginCommands();_graph.RecordUpload(0,(uint)(768+right.Length));
            _graph.Record(op,0,768,0,(uint)left.Length);_graph.RecordDownload(0,(uint)left.Length);
            _graph.Context.SubmitAndWait();_graph.Readback.Read(output);
        }
    }
    public void Dispose() => _graph.Dispose();
}

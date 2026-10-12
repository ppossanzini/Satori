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
    public void Dispose() => _graph.Dispose();
}

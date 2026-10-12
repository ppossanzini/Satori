using BenchmarkDotNet.Attributes;
using Satori.Formats.Gguf;

namespace Satori.Benchmarks;

[MemoryDiagnoser]
public class GgufBenchmarks
{
    public static string ModelDirectory { get; set; } = "";
    [Params("q8_0", "q4_0")] public string Format { get; set; } = "";
    [Params("blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight")]
    public string Tensor { get; set; } = "";
    private GgufFile _file = null!;
    private GgufQuantizedMatrix _matrix = null!;
    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        _file = new GgufFile(Path.Combine(ModelDirectory, $"stories15M-{Format}.gguf"));
        _matrix = new(_file, Tensor);
        _input = Enumerable.Range(0, _matrix.Shape.Columns).Select(i => (i * 17 % 31 - 15) / 16f).ToArray();
        _output = new float[_matrix.Shape.Rows];
    }

    [Benchmark(Baseline = true)] public void Scalar() => _matrix.ExecuteScalar(_input, _output);
    [Benchmark] public void Avx2OrScalar() => _matrix.Execute(_input, _output);
    [GlobalCleanup] public void Cleanup() => _file?.Dispose();
}

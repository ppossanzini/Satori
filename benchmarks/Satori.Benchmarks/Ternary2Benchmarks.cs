using BenchmarkDotNet.Attributes;
using Satori.Core;
using Satori.Validation;

namespace Satori.Benchmarks;

[MemoryDiagnoser]
public class Ternary2Benchmarks
{
    [Params(1024, 2048, 4096)]
    public int Size { get; set; }

    [Params(64, 128, 256)]
    public int BlockSize { get; set; }

    private Ternary2Matrix _matrix = null!;
    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        var sample = NumericalCorpus.RandomMatrix(Size, Size);
        _matrix = Ternary2Matrix.Quantize(sample.Weights, Size, Size, BlockSize);
        _input = sample.Input;
        _output = new float[Size];
    }

    [Benchmark(Baseline = true)]
    public void Scalar() => Ternary2MatVec.ExecuteScalar(_matrix, _input, _output);

    [Benchmark]
    public void Avx2OrScalar() => Ternary2MatVec.ExecuteAvx2(_matrix, _input, _output);
}

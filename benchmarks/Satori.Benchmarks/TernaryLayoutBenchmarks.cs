using BenchmarkDotNet.Attributes;
using Satori.Core;
using Satori.Validation;

namespace Satori.Benchmarks;

[MemoryDiagnoser]
public class TernaryLayoutBenchmarks
{
    [Params(1024, 2048, 4096)] public int Size { get; set; }
    [Params(64, 128, 256)] public int BlockSize { get; set; }
    private Ternary2Matrix _ternary = null!;
    private TernaryBitplaneMatrix _bitplane = null!;
    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        var sample = NumericalCorpus.RandomMatrix(Size, Size);
        _ternary = Ternary2Matrix.Quantize(sample.Weights, Size, Size, BlockSize);
        _bitplane = TernaryBitplaneMatrix.FromTernary2(_ternary);
        _input = sample.Input;
        _output = new float[Size];
        float[] expected = new float[Size];
        Ternary2MatVec.ExecuteScalar(_ternary, _input, expected);
        TernaryBitplaneMatVec.ExecuteScalar(_bitplane, _input, _output);
        if (!expected.SequenceEqual(_output)) throw new InvalidDataException("Layout scalar results differ.");
    }

    [Benchmark(Baseline = true)] public void Ternary2Scalar() => Ternary2MatVec.ExecuteScalar(_ternary, _input, _output);
    [Benchmark] public void Ternary2Auto() => Ternary2MatVec.Execute(_ternary, _input, _output);
    [Benchmark] public void BitplaneAuto() => TernaryBitplaneMatVec.Execute(_bitplane, _input, _output);
    [Benchmark] public void BitplaneScalar() => TernaryBitplaneMatVec.ExecuteScalar(_bitplane, _input, _output);
}

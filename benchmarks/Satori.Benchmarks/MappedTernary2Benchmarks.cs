using BenchmarkDotNet.Attributes;
using Satori.Core;
using Satori.Validation;

namespace Satori.Benchmarks;

[MemoryDiagnoser]
public class MappedTernary2Benchmarks
{
    [Params(1024, 2048, 4096)]
    public int Size { get; set; }

    [Params(64)]
    public int BlockSize { get; set; }

    private string _directory = null!;
    private Ternary2Matrix _owned = null!;
    private MappedTernary2Matrix _mapped = null!;
    private float[] _input = null!;
    private float[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        _directory = Path.GetFullPath(Path.Combine(".temp", "t10", Guid.NewGuid().ToString("N")));
        Directory.CreateDirectory(_directory);
        try
        {
            var sample = NumericalCorpus.RandomMatrix(Size, Size);
            _owned = Ternary2Matrix.Quantize(sample.Weights, Size, Size, BlockSize);
            string path = Path.Combine(_directory, "matrix.stq2");
            _owned.Save(path);
            _mapped = new MappedTernary2Matrix(path);
            _input = sample.Input;
            _output = new float[Size];
        }
        catch
        {
            _mapped?.Dispose();
            Directory.Delete(_directory, recursive: true);
            throw;
        }
    }

    [Benchmark(Baseline = true)]
    public void OwnedAuto() => Ternary2MatVec.Execute(_owned, _input, _output);

    [Benchmark]
    public void MappedAuto() => _mapped.Execute(_input, _output);

    [GlobalCleanup]
    public void Cleanup()
    {
        _mapped?.Dispose();
        Directory.Delete(_directory, recursive: true);
    }
}

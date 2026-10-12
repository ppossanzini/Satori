using System.Diagnostics;
using Satori.Core;
using Satori.Validation;

namespace Satori.Benchmarks;

internal static class MappedStartupProbe
{
    internal static List<MappedStartupResult> Run()
    {
        var results = new List<MappedStartupResult>();
        string directory = Path.GetFullPath(Path.Combine(".temp", "t10", Guid.NewGuid().ToString("N")));
        Directory.CreateDirectory(directory);
        try
        {
            foreach (int size in new[] { 1024, 2048, 4096 })
            {
                var sample = NumericalCorpus.RandomMatrix(size, size);
                var owned = Ternary2Matrix.Quantize(sample.Weights, size, size, 64);
                string path = Path.Combine(directory, $"{size}.stq2");
                owned.Save(path);
                float[] expected = new float[size], actual = new float[size];
                // Warm the shared compute code independently of the mapped first call.
                for (int i = 0; i < 5; i++) Ternary2MatVec.Execute(owned, sample.Input, expected);
                for (int trial = 1; trial <= 3; trial++)
                {
                    var watch = Stopwatch.StartNew();
                    using var mapped = new MappedTernary2Matrix(path);
                    double open = watch.Elapsed.TotalMilliseconds;
                    watch.Restart();
                    mapped.Execute(sample.Input, actual);
                    double first = watch.Elapsed.TotalMicroseconds;
                    results.Add(new(size, 64, trial, open, first, expected.AsSpan().SequenceEqual(actual)));
                }
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
        return results;
    }
}

using System.Text.Json;
using Satori.Core;

namespace Satori.Validation;

public static class TernaryLayoutValidation
{
    public static bool Run(string reportPath)
    {
        var results = new List<object>();
        bool passed = true;
        foreach (int blockSize in new[] { 64, 128, 256 })
        {
            var samples = NumericalCorpus.CriticalCases(blockSize).Concat(
                new[] { 1024, 2048, 4096 }.Select(size => NumericalCorpus.RandomMatrix(size, size)));
            foreach (var sample in samples)
            {
                var original = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns, blockSize);
                var converted = TernaryBitplaneMatrix.FromTernary2(original);
                float[] scalar = new float[sample.Rows], simd = new float[sample.Rows], reference = new float[sample.Rows];
                Ternary2MatVec.ExecuteScalar(original, sample.Input, scalar);
                TernaryBitplaneMatVec.ExecuteScalar(converted, sample.Input, reference);
                bool scalarEqual = scalar.SequenceEqual(reference);
                Ternary2MatVec.Execute(original, sample.Input, simd);
                TernaryBitplaneMatVec.Execute(converted, sample.Input, reference);
                bool simdEqual = simd.SequenceEqual(reference);
                var fp64 = NumericalValidation.Evaluate(sample, blockSize, useAvx2: true);
                bool valid = scalarEqual && simdEqual && fp64.Passed;
                passed &= valid;
                results.Add(new { sample.Name, sample.Rows, sample.Columns, BlockSize = blockSize,
                    converted.ByteLength, BytesPerWeight = (double)converted.ByteLength / ((long)sample.Rows * sample.Columns),
                    ScalarEqual = scalarEqual, AutoEqual = simdEqual, Fp64MaximumErrorOverBound = fp64.MaxErrorToBoundRatio, Passed = valid });
            }
        }
        File.WriteAllText(reportPath, JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }));
        return passed;
    }
}

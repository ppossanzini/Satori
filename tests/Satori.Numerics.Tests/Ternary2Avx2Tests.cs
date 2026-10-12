using Satori.Core;
using Satori.Validation;

namespace Satori.Numerics.Tests;

public class Ternary2Avx2Tests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void VectorKernelMatchesReferenceIncludingTailsAndExtremeScales(int blockSize)
    {
        foreach (var sample in NumericalCorpus.CriticalCases(blockSize))
            Assert.True(NumericalValidation.Evaluate(sample, blockSize, useAvx2: true).Passed, sample.Name);

        foreach (int columns in Enumerable.Range(1, 17).Concat(new[] { 63, 64, 65, 127, 128, 129, 255, 256, 257 }))
        {
            var result = NumericalValidation.Evaluate(NumericalCorpus.RandomMatrix(3, columns), blockSize, useAvx2: true);
            Assert.True(result.Passed, $"columns={columns}: ratio={result.MaxErrorToBoundRatio}");
        }
    }

    [Fact]
    public void VectorPathAllocatesNoManagedMemoryAfterWarmup()
    {
        var sample = NumericalCorpus.RandomMatrix(7, 257);
        var matrix = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns);
        float[] output = new float[sample.Rows];
        for (int i = 0; i < 100; i++) Ternary2MatVec.ExecuteAvx2(matrix, sample.Input, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) Ternary2MatVec.ExecuteAvx2(matrix, sample.Input, output);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}

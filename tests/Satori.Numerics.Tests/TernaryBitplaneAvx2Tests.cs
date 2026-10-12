using Satori.Core;
using Satori.Validation;

namespace Satori.Numerics.Tests;

public sealed class TernaryBitplaneAvx2Tests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void SameCorpusAndSimdOrderAsTernary2IncludingTails(int blockSize)
    {
        var cases = NumericalCorpus.CriticalCases(blockSize).Concat(
            new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257 }
                .Select(columns => NumericalCorpus.RandomMatrix(3, columns)));
        foreach (var sample in cases)
        {
            var ternary = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns, blockSize);
            var bitplane = TernaryBitplaneMatrix.FromTernary2(ternary);
            float[] expected = new float[sample.Rows], output = new float[sample.Rows];
            Ternary2MatVec.Execute(ternary, sample.Input, expected);
            TernaryBitplaneMatVec.Execute(bitplane, sample.Input, output);
            Assert.Equal(expected, output);
            float[] reconstructed = new float[sample.Weights.Length];
            ternary.Dequantize(reconstructed);
            int blocks = (sample.Columns + blockSize - 1) / blockSize;
            double n = blockSize + blocks + 2;
            double gamma = n * Math.ScaleB(1d, -24) / (1 - n * Math.ScaleB(1d, -24));
            for (int row = 0; row < sample.Rows; row++)
            {
                double reference = 0, magnitude = 0;
                for (int col = 0; col < sample.Columns; col++)
                {
                    double value = (double)reconstructed[row * sample.Columns + col] * sample.Input[col];
                    reference += value; magnitude += Math.Abs(value);
                }
                Assert.InRange(Math.Abs(output[row] - reference), 0, gamma * magnitude + (sample.Columns + blocks) * (double)float.Epsilon);
            }
            for (int i = 0; i < 20; i++) TernaryBitplaneMatVec.Execute(bitplane, sample.Input, output);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 20; i++) TernaryBitplaneMatVec.Execute(bitplane, sample.Input, output);
            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        }
    }
}

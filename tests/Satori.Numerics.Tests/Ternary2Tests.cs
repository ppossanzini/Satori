using System.Buffers.Binary;
using Satori.Core;

namespace Satori.Numerics.Tests;

public class Ternary2Tests
{
    [Fact]
    public void EncodingMatchesIndependentBinaryFixture()
    {
        float[] source = new float[64];
        source[0] = 2;
        source[1] = -2;
        var matrix = Ternary2Matrix.Quantize(source, 1, 64);
        byte[] actual = new byte[matrix.ByteLength];
        matrix.CopyDataTo(actual);
        byte[] expected = new byte[20];
        expected[2] = 0x80;
        expected[3] = 0x3D;
        expected[4] = 0x09;
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ImportRejectsCorruptBlocks()
    {
        byte[] payload = new byte[20];
        BinaryPrimitives.WriteSingleLittleEndian(payload, 1);
        payload[4] = 3;
        Assert.Throws<ArgumentException>(() => Ternary2Matrix.FromPacked(1, 1, 64, payload));
        payload[4] = 4; // Nonzero padding after the single valid weight.
        Assert.Throws<ArgumentException>(() => Ternary2Matrix.FromPacked(1, 1, 64, payload));
        payload[4] = 1;
        BinaryPrimitives.WriteSingleLittleEndian(payload, 0);
        Assert.Throws<ArgumentException>(() => Ternary2Matrix.FromPacked(1, 1, 64, payload));
        payload[4] = 0;
        foreach (float scale in new[] { float.NaN, float.PositiveInfinity, -1f, -0f })
        {
            BinaryPrimitives.WriteSingleLittleEndian(payload, scale);
            Assert.Throws<ArgumentException>(() => Ternary2Matrix.FromPacked(1, 1, 64, payload));
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void PackedMatVecMatchesDoubleReferenceAcrossRowAndBlockBoundaries(int blockSize)
    {
        const int rows = 7;
        int columns = blockSize * 2 + 3;
        var random = new Random(1729);
        float[] weights = new float[rows * columns];
        float[] input = new float[columns];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(random.NextDouble() * 4 - 2);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        // Include an all-zero row adjacent to nonzero rows.
        weights.AsSpan(columns, columns).Clear();
        var matrix = Ternary2Matrix.Quantize(weights, rows, columns, blockSize);
        byte[] packed = new byte[matrix.ByteLength];
        matrix.CopyDataTo(packed);
        matrix = Ternary2Matrix.FromPacked(rows, columns, blockSize, packed);
        float[] output = new float[rows];
        Ternary2MatVec.Execute(matrix, input, output);
        int blocks = (columns + blockSize - 1) / blockSize;
        int stride = 4 + blockSize / 4;
        for (int row = 0; row < rows; row++)
        {
            double reference = 0;
            double absoluteSum = 0;
            for (int column = 0; column < columns; column++)
            {
                int offset = (row * blocks + column / blockSize) * stride;
                double scale = BinaryPrimitives.ReadSingleLittleEndian(packed.AsSpan(offset));
                int index = column % blockSize;
                int code = (packed[offset + 4 + index / 4] >> (index % 4 * 2)) & 3;
                double term = (code == 1 ? scale : code == 2 ? -scale : 0) * input[column];
                reference += term;
                absoluteSum += Math.Abs(term);
            }
            Assert.True(Math.Abs(output[row] - reference) <= 2e-6 * absoluteSum + 1e-7);
        }
        for (int i = 0; i < 100; i++) Ternary2MatVec.Execute(matrix, input, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) Ternary2MatVec.Execute(matrix, input, output);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void QuantizationHandlesExtremeFiniteValuesAndRejectsNonFiniteValues()
    {
        foreach (float value in new[] { float.MaxValue, float.Epsilon, 0f })
        {
            var matrix = Ternary2Matrix.Quantize([value, -value], 1, 2);
            float[] reconstructed = new float[2];
            matrix.Dequantize(reconstructed);
            Assert.Equal(new[] { value, -value }, reconstructed);
        }
        foreach (float value in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
            Assert.Throws<ArgumentException>(() => Ternary2Matrix.Quantize([value], 1, 1));
    }
}

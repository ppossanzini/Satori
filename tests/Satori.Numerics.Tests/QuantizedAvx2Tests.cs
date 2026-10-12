using System.Buffers.Binary;
using Satori.Core;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class QuantizedAvx2Tests
{
    [Theory]
    [InlineData(BlockQuantization.Q8_0)]
    [InlineData(BlockQuantization.Q4_0)]
    public void SimdOrFallbackMatchesIndependentFp64AndDoesNotAllocate(BlockQuantization format)
    {
        var random = new Random(1729);
        foreach (int columns in new[] { 32, 64, 96, 288, 768 })
        {
            var shape = new QuantizedMatrixShape(7, columns, format);
            var bytes = new byte[shape.ByteLength];
            var input = Enumerable.Range(0, columns).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var output = new float[shape.Rows];
            var expected = new double[shape.Rows];
            var magnitude = new double[shape.Rows];
            for (int r = 0; r < shape.Rows; r++)
            for (int b = 0; b < columns / 32; b++)
            {
                int offset = (r * (columns / 32) + b) * shape.BlockBytes;
                Half scale = r switch { 0 => (Half)0, 1 => (Half)(-0f), 2 => Half.Epsilon, 3 => (Half)65504, _ => (Half)(random.NextDouble() * 4 - 2) };
                BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits(scale));
                random.NextBytes(bytes.AsSpan(offset + 2, shape.BlockBytes - 2));
                for (int j = 0; j < 32; j++)
                {
                    int code = format == BlockQuantization.Q8_0 ? (sbyte)bytes[offset + 2 + j] :
                        ((bytes[offset + 2 + j % 16] >> (j < 16 ? 0 : 4)) & 15) - 8;
                    double product = (double)(float)scale * code * input[b * 32 + j];
                    expected[r] += product;
                    magnitude[r] += Math.Abs(product);
                }
            }
            var matrix = new QuantizedMatrix(bytes, shape);
            matrix.Execute(input, output);
            double n = columns + columns / 32 + 2;
            double gamma = n * Math.ScaleB(1.0, -24) / (1 - n * Math.ScaleB(1.0, -24));
            for (int r = 0; r < shape.Rows; r++)
                Assert.InRange(Math.Abs(output[r] - expected[r]), 0, gamma * magnitude[r] + n * float.Epsilon);
            for (int i = 0; i < 20; i++) matrix.Execute(input, output);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 20; i++) matrix.Execute(input, output);
            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        }
    }

    [Theory]
    [InlineData("q8_0")]
    [InlineData("q4_0")]
    public void MappedGoldenAutoIsExactAndAllocationFree(string format)
    {
        using var file = new GgufFile(Path.Combine(GgufTests.Root, "fixtures/gguf/v1", format + ".gguf"));
        var matrix = new GgufQuantizedMatrix(file, "weights");
        float[] input = Enumerable.Repeat(1f, 32).ToArray(), output = new float[1];
        for (int i = 0; i < 20; i++) matrix.Execute(input, output);
        Assert.Equal(-8f, output[0]);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) matrix.Execute(input, output);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }
}

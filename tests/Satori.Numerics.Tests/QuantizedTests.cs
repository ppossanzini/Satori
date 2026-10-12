using System.Buffers.Binary;
using Satori.Core;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class QuantizedTests
{
    [Theory]
    [InlineData("q8_0", 34)]
    [InlineData("q4_0", 18)]
    public void GoldenImportedAndOwnedAgreeWithManualDot(string format, int length)
    {
        using var file = new GgufFile(Path.Combine(GgufTests.Root, "fixtures/gguf/v1/" + format + ".gguf"));
        var mapped = new GgufQuantizedMatrix(file, "weights");
        float[] input = Enumerable.Repeat(1f, 32).ToArray(), output = new float[1];
        mapped.ExecuteScalar(input, output);
        Assert.Equal(-8f, output[0]);
        file.ReadTensor("weights", 0, length, (bytes, _) =>
        {
            var owned = new QuantizedMatrix(bytes, mapped.Shape);
            owned.ExecuteScalar(input, output);
            Assert.Equal(-8f, output[0]);
        }, 0);
    }

    [Fact]
    public void Q8FullSignedRangeNegativeSubnormalAndZeroScalesMatchFp64()
    {
        var shape = new QuantizedMatrixShape(7, 96, BlockQuantization.Q8_0);
        byte[] bytes = new byte[shape.ByteLength];
        float[] scales = [0.5f, -2f, 0f, -0f, (float)Half.Epsilon, 65504f, 1f];
        float[] input = Enumerable.Range(0, 96).Select(i => (i % 13 - 6) / 16f).ToArray();
        float[] output = new float[7];
        double[] reference = new double[7];
        for (int row = 0; row < 7; row++)
        for (int b = 0; b < 3; b++)
        {
            int offset = (row * 3 + b) * 34;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)scales[row]));
            for (int j = 0; j < 32; j++)
            {
                int q = (row * 32 + b * 17 + j * 7) % 256 - 128;
                bytes[offset + 2 + j] = unchecked((byte)(sbyte)q);
                reference[row] += (double)scales[row] * q * input[b * 32 + j];
            }
        }
        var matrix = new QuantizedMatrix(bytes, shape);
        matrix.ExecuteScalar(input, output);
        for (int row = 0; row < 7; row++) Assert.InRange(Math.Abs(output[row] - reference[row]), 0, 0.001);
        for (int i = 0; i < 20; i++) matrix.ExecuteScalar(input, output);
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) matrix.ExecuteScalar(input, output);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - start);
    }

    [Theory]
    [InlineData((ushort)0x7c00)]
    [InlineData((ushort)0xfc00)]
    [InlineData((ushort)0x7e00)]
    public void NonfiniteScalesRejectBeforeCompute(ushort scale)
    {
        byte[] data = new byte[34];
        BinaryPrimitives.WriteUInt16LittleEndian(data, scale);
        Assert.Throws<InvalidDataException>(() => new QuantizedMatrix(data, new(1, 32, BlockQuantization.Q8_0)));
        byte[] golden = File.ReadAllBytes(Path.Combine(GgufTests.Root, "fixtures/gguf/v1/q8_0.gguf"));
        BinaryPrimitives.WriteUInt16LittleEndian(golden.AsSpan(160), scale);
        using var temp = new MappedFileFixture(golden.Length, 0, golden);
        using var file = new GgufFile(temp.Path);
        Assert.Throws<InvalidDataException>(() => new GgufQuantizedMatrix(file, "weights"));
    }
}

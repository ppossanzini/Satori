using System.Buffers.Binary;
using Satori.Core;

namespace Satori.Numerics.Tests;

public sealed class Q4Tests
{
    [Fact]
    public void AllNibblesBothHalvesAndSignedScalesMatchIndependentFp64()
    {
        var shape = new QuantizedMatrixShape(6, 96, BlockQuantization.Q4_0);
        byte[] bytes = new byte[shape.ByteLength];
        float[] scales = [0.5f, -2f, 0f, -0f, (float)Half.Epsilon, 65504f];
        float[] input = Enumerable.Range(0, 96).Select(i => (i % 17 - 8) / 32f).ToArray();
        float[] output = new float[shape.Rows];
        double[] expected = new double[shape.Rows];
        for (int row = 0; row < shape.Rows; row++)
        for (int b = 0; b < 3; b++)
        {
            int offset = (row * 3 + b) * 18;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)scales[row]));
            for (int j = 0; j < 16; j++)
            {
                int lo = (j + b) % 16, hi = (15 - j + b) % 16;
                bytes[offset + 2 + j] = (byte)(lo | hi << 4);
                expected[row] += (double)scales[row] * (lo - 8) * input[b * 32 + j];
                expected[row] += (double)scales[row] * (hi - 8) * input[b * 32 + j + 16];
            }
        }
        var matrix = new QuantizedMatrix(bytes, shape);
        matrix.ExecuteScalar(input, output);
        for (int row = 0; row < shape.Rows; row++) Assert.InRange(Math.Abs(output[row] - expected[row]), 0, 0.001);
        for (int i = 0; i < 20; i++) matrix.ExecuteScalar(input, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) matrix.ExecuteScalar(input, output);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Theory]
    [InlineData((ushort)0x7c00)]
    [InlineData((ushort)0x7e00)]
    public void NonfiniteQ4ScaleRejects(ushort scale)
    {
        byte[] data = new byte[18];
        BinaryPrimitives.WriteUInt16LittleEndian(data, scale);
        Assert.Throws<InvalidDataException>(() => new QuantizedMatrix(data, new(1, 32, BlockQuantization.Q4_0)));
    }
}

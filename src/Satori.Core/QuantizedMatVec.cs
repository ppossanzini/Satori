using System.Buffers.Binary;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

public static class QuantizedMatVec
{
    /// <summary>Selects AVX2 when available, otherwise scalar, on previously validated immutable weights.</summary>
    public static void Execute(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (Avx2.IsSupported) ExecuteAvx2(data, shape, input, output);
        else ExecuteScalar(data, shape, input, output);
    }

    public static void ExecuteAvx2(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("AVX2 is unavailable.");
        Guard(data, shape, input, output);
        QuantizedAvx2Kernel.Execute(data, shape, input, output);
    }

    public static void ValidatePayload(ReadOnlySpan<byte> data, QuantizedMatrixShape shape)
    {
        if (data.Length != shape.ByteLength) throw new ArgumentException("Incorrect quantized payload length.", nameof(data));
        for (int offset = 0; offset < data.Length; offset += shape.BlockBytes)
            if (!float.IsFinite(Scale(data[offset..]))) throw new InvalidDataException($"Nonfinite block scale at {offset}.");
    }

    /// <summary>Multiplies immutable compressed weights whose scales have already been validated.</summary>
    public static void ExecuteScalar(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        Guard(data, shape, input, output);
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int column = 0; column < shape.Columns; column += 32)
            {
                int offset = (row * (shape.Columns / 32) + column / 32) * shape.BlockBytes;
                var block = data.Slice(offset, shape.BlockBytes);
                float sum = 0;
                if (shape.Format == BlockQuantization.Q8_0)
                    for (int j = 0; j < 32; j++) sum += (sbyte)block[2 + j] * input[column + j];
                else
                {
                    for (int j = 0; j < 16; j++) sum += ((block[2 + j] & 15) - 8) * input[column + j];
                    for (int j = 0; j < 16; j++) sum += ((block[2 + j] >> 4) - 8) * input[column + 16 + j];
                }
                total += Scale(block) * sum;
            }
            output[row] = total;
        }
    }

    internal static void Guard(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (data.Length != shape.ByteLength || input.Length != shape.Columns || output.Length != shape.Rows)
            throw new ArgumentException("Matrix/vector dimensions do not match.");
        if (input.Overlaps(output)) throw new ArgumentException("Input and output must not overlap.");
    }

    internal static float Scale(ReadOnlySpan<byte> block) => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
}

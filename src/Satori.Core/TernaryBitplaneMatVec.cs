using System.Buffers.Binary;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

public static class TernaryBitplaneMatVec
{
    /// <summary>Uses AVX2 when supported, otherwise scalar.</summary>
    public static void Execute(TernaryBitplaneMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (Avx2.IsSupported) ExecuteAvx2(matrix, input, output);
        else ExecuteScalar(matrix, input, output);
    }

    public static void ExecuteAvx2(TernaryBitplaneMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("AVX2 is unavailable.");
        Guard(matrix, input, output);
        TernaryBitplaneAvx2Kernel.Execute(matrix, input, output);
    }

    public static void ExecuteScalar(TernaryBitplaneMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        Guard(matrix, input, output);
        var shape = matrix.Shape;
        int planeBytes = shape.BlockSize / 8;
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int b = 0; b < shape.BlocksPerRow; b++)
            {
                var bytes = matrix.Data.Slice((row * shape.BlocksPerRow + b) * shape.BlockByteLength, shape.BlockByteLength);
                int column = b * shape.BlockSize;
                int count = Math.Min(shape.BlockSize, shape.Columns - column);
                float sum = 0;
                for (int i = 0; i < count; i++)
                {
                    int mask = 1 << (i % 8);
                    if ((bytes[4 + i / 8] & mask) != 0) sum += input[column + i];
                    else if ((bytes[4 + planeBytes + i / 8] & mask) != 0) sum -= input[column + i];
                }
                total += BinaryPrimitives.ReadSingleLittleEndian(bytes) * sum;
            }
            output[row] = total;
        }
    }

    internal static void Guard(TernaryBitplaneMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != matrix.Columns || output.Length != matrix.Rows)
            throw new ArgumentException("Matrix/vector dimensions do not match.");
        if (input.Overlaps(output)) throw new ArgumentException("Input and output must not overlap.");
    }
}

using System.Buffers.Binary;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

/// <summary>CPU MatVec directly over packed ternary blocks.</summary>
public static class Ternary2MatVec
{
    public static void Execute(Ternary2Matrix matrix, ReadOnlySpan<float> input, Span<float> output)
        => ExecuteAvx2(matrix, input, output);

    /// <summary>Uses AVX2 when supported, otherwise the scalar kernel.</summary>
    public static void ExecuteAvx2(Ternary2Matrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Avx2.IsSupported)
        {
            ExecuteScalar(matrix, input, output);
            return;
        }
        Validate(matrix, input, output);
        Ternary2Avx2Kernel.Execute(matrix.Data, matrix.Shape, input, output);
    }

    internal static void Validate(Ternary2Matrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        ValidateVectors(matrix.Shape, input, output);
    }

    internal static void ValidateVectors(Ternary2Shape matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != matrix.Columns || output.Length != matrix.Rows)
            throw new ArgumentException("Vector lengths must match matrix dimensions.");
        if (input.Overlaps(output))
            throw new ArgumentException("Output must not overlap input.", nameof(output));
    }

    public static void ExecuteScalar(Ternary2Matrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        Validate(matrix, input, output);
        ExecuteScalarCore(matrix.Data, matrix.Shape, input, output);
    }

    internal static void ExecuteData(ReadOnlySpan<byte> data, Ternary2Execution execution)
    {
        if (execution.UseAvx2 && Avx2.IsSupported)
            Ternary2Avx2Kernel.Execute(data, execution.Shape, execution.Input, execution.Output);
        else
            ExecuteScalarCore(data, execution.Shape, execution.Input, execution.Output);
    }

    private static void ExecuteScalarCore(ReadOnlySpan<byte> data, Ternary2Shape matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        int blocks = matrix.BlocksPerRow;
        int stride = matrix.BlockByteLength;
        for (int row = 0; row < matrix.Rows; row++)
        {
            float sum = 0;
            for (int block = 0; block < blocks; block++)
            {
                var bytes = data.Slice((row * blocks + block) * stride, stride);
                float scale = BinaryPrimitives.ReadSingleLittleEndian(bytes);
                int column = block * matrix.BlockSize;
                int count = Math.Min(matrix.BlockSize, matrix.Columns - column);
                float partial = 0;
                for (int i = 0; i < count; i++)
                {
                    int code = (bytes[4 + i / 4] >> (2 * (i % 4))) & 3;
                    if (code == 1) partial += input[column + i];
                    else if (code == 2) partial -= input[column + i];
                }
                sum += partial * scale;
            }
            output[row] = sum;
        }
    }
}

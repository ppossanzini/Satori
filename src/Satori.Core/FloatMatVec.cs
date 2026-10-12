namespace Satori.Core;

/// <summary>Scalar FP32 matrix-vector product with row-major weights.</summary>
public static class FloatMatVec
{
    public static void Execute(
        ReadOnlySpan<float> weights,
        int rows,
        int columns,
        ReadOnlySpan<float> input,
        Span<float> output)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (weights.Length != (long)rows * columns)
            throw new ArgumentException("Weights must match the matrix dimensions.", nameof(weights));
        if (input.Length != columns)
            throw new ArgumentException("Input length must match columns.", nameof(input));
        if (output.Length != rows)
            throw new ArgumentException("Output length must match rows.", nameof(output));
        if (input.Overlaps(output) || weights.Overlaps(output))
            throw new ArgumentException("Output must not overlap input or weights.", nameof(output));

        for (int row = 0; row < rows; row++)
        {
            float sum = 0;
            int offset = row * columns;
            for (int column = 0; column < columns; column++)
                sum += weights[offset + column] * input[column];
            output[row] = sum;
        }
    }
}

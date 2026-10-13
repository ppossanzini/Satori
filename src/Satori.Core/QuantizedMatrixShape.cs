namespace Satori.Core;

public sealed record QuantizedMatrixShape
{
    public QuantizedMatrixShape(int rows, int columns, BlockQuantization format)
    {
        (BlockElements, BlockBytes) = format switch
        {
            BlockQuantization.Q8_0 => (32, 34),
            BlockQuantization.Q4_0 => (32, 18),
            BlockQuantization.Q4_K => (Q4KCodec.ElementsPerBlock, Q4KCodec.BytesPerBlock),
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };
        if (rows <= 0 || columns <= 0 || columns % BlockElements != 0)
            throw new ArgumentOutOfRangeException(nameof(columns),
                $"Positive dimensions and complete {BlockElements}-weight rows required.");
        Rows = rows;
        Columns = columns;
        Format = format;
        ByteLength = checked(rows * (columns / BlockElements) * BlockBytes);
    }

    public int Rows { get; }
    public int Columns { get; }
    public BlockQuantization Format { get; }

    /// <summary>Weights covered by one block of this format.</summary>
    public int BlockElements { get; }

    public int BlockBytes { get; }
    public int ByteLength { get; }
}

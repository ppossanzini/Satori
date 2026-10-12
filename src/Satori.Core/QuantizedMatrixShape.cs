namespace Satori.Core;

public sealed record QuantizedMatrixShape
{
    public QuantizedMatrixShape(int rows, int columns, BlockQuantization format)
    {
        if (rows <= 0 || columns <= 0 || columns % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(columns), "Positive dimensions and complete 32-weight rows required.");
        if (format is not (BlockQuantization.Q8_0 or BlockQuantization.Q4_0)) throw new ArgumentOutOfRangeException(nameof(format));
        Rows = rows;
        Columns = columns;
        Format = format;
        BlockBytes = format == BlockQuantization.Q8_0 ? 34 : 18;
        ByteLength = checked(rows * (columns / 32) * BlockBytes);
    }

    public int Rows { get; }
    public int Columns { get; }
    public BlockQuantization Format { get; }
    public int BlockBytes { get; }
    public int ByteLength { get; }
}

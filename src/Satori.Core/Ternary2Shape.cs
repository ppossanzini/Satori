namespace Satori.Core;

internal readonly record struct Ternary2Shape
{
    internal Ternary2Shape(int rows, int columns, int blockSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (blockSize is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(blockSize), "Supported block sizes: 64, 128, 256.");
        Rows = rows;
        Columns = columns;
        BlockSize = blockSize;
        ByteLength = checked((int)(rows * (((long)columns + blockSize - 1) / blockSize) * (4 + blockSize / 4)));
    }

    internal int Rows { get; }
    internal int Columns { get; }
    internal int BlockSize { get; }
    internal int ByteLength { get; }
    internal int BlocksPerRow => (int)(((long)Columns + BlockSize - 1) / BlockSize);
    internal int BlockByteLength => 4 + BlockSize / 4;
}

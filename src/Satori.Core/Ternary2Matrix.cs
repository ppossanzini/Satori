using System.Buffers.Binary;

namespace Satori.Core;

/// <summary>Owned, validated row-major ternary weights with FP32 block scales.</summary>
public sealed class Ternary2Matrix
{
    private readonly byte[] _data;

    private Ternary2Matrix(int rows, int columns, int blockSize, byte[] data)
    {
        Rows = rows;
        Columns = columns;
        BlockSize = blockSize;
        _data = data;
    }

    public int Rows { get; }
    public int Columns { get; }
    public int BlockSize { get; }
    public int ByteLength => _data.Length;
    internal int BlocksPerRow => (int)(((long)Columns + BlockSize - 1) / BlockSize);
    internal int BlockByteLength => 4 + BlockSize / 4;
    internal ReadOnlySpan<byte> Data => _data;
    internal Ternary2Shape Shape => new(Rows, Columns, BlockSize);

    public void CopyDataTo(Span<byte> destination) => _data.AsSpan().CopyTo(destination);

    public void Save(string path)
    {
        Span<byte> header = stackalloc byte[Ternary2FileFormat.HeaderSize];
        Ternary2FileFormat.WriteHeader(header, Shape);
        using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        file.Write(header);
        file.Write(_data);
    }

    public static Ternary2Matrix Quantize(ReadOnlySpan<float> weights, int rows, int columns, int blockSize = 64)
    {
        int length = GetByteLength(rows, columns, blockSize);
        if (weights.Length != (long)rows * columns)
            throw new ArgumentException("Weights must match matrix dimensions.", nameof(weights));
        foreach (float value in weights)
            if (!float.IsFinite(value))
                throw new ArgumentException("Weights must be finite.", nameof(weights));

        byte[] data = new byte[length];
        var matrix = new Ternary2Matrix(rows, columns, blockSize, data);
        for (int row = 0; row < rows; row++)
        for (int block = 0; block < matrix.BlocksPerRow; block++)
        {
            int column = block * blockSize;
            int count = Math.Min(blockSize, columns - column);
            var source = weights.Slice(row * columns + column, count);
            double magnitude = 0;
            foreach (float value in source)
                magnitude += Math.Abs((double)value);
            float scale = (float)(magnitude / count);
            var target = data.AsSpan((row * matrix.BlocksPerRow + block) * matrix.BlockByteLength, matrix.BlockByteLength);
            BinaryPrimitives.WriteSingleLittleEndian(target, scale);
            if (scale == 0)
                continue;
            double threshold = (double)scale / 2;
            for (int i = 0; i < count; i++)
            {
                int code = source[i] > threshold ? 1 : source[i] < -threshold ? 2 : 0;
                target[4 + i / 4] |= (byte)(code << (2 * (i % 4)));
            }
        }
        return matrix;
    }

    public static Ternary2Matrix FromPacked(int rows, int columns, int blockSize, ReadOnlySpan<byte> data)
    {
        if (data.Length != GetByteLength(rows, columns, blockSize))
            throw new ArgumentException("Payload length must match matrix dimensions.", nameof(data));
        byte[] owned = data.ToArray();
        var matrix = new Ternary2Matrix(rows, columns, blockSize, owned);
        ValidatePayload(matrix.Shape, owned);
        return matrix;
    }

    internal static void ValidatePayload(Ternary2Shape shape, ReadOnlySpan<byte> data)
    {
        int rows = shape.Rows, columns = shape.Columns, blockSize = shape.BlockSize;
        for (int row = 0; row < rows; row++)
        for (int block = 0; block < shape.BlocksPerRow; block++)
        {
            var bytes = data.Slice((row * shape.BlocksPerRow + block) * shape.BlockByteLength, shape.BlockByteLength);
            float scale = BinaryPrimitives.ReadSingleLittleEndian(bytes);
            if (!float.IsFinite(scale) || scale < 0 || BitConverter.SingleToInt32Bits(scale) == int.MinValue)
                throw new ArgumentException("Invalid block scale.", nameof(data));
            int count = Math.Min(blockSize, columns - block * blockSize);
            for (int i = 0; i < blockSize; i++)
            {
                int code = (bytes[4 + i / 4] >> (2 * (i % 4))) & 3;
                if (code == 3 || ((i >= count || scale == 0) && code != 0))
                    throw new ArgumentException("Invalid weight code or padding.", nameof(data));
            }
        }
    }

    public void Dequantize(Span<float> destination)
    {
        if (destination.Length != (long)Rows * Columns)
            throw new ArgumentException("Destination must match matrix dimensions.", nameof(destination));
        for (int row = 0; row < Rows; row++)
        for (int block = 0; block < BlocksPerRow; block++)
        {
            var bytes = Data.Slice((row * BlocksPerRow + block) * BlockByteLength, BlockByteLength);
            float scale = BinaryPrimitives.ReadSingleLittleEndian(bytes);
            int column = block * BlockSize;
            int count = Math.Min(BlockSize, Columns - column);
            for (int i = 0; i < count; i++)
            {
                int code = (bytes[4 + i / 4] >> (2 * (i % 4))) & 3;
                destination[row * Columns + column + i] = code == 1 ? scale : code == 2 ? -scale : 0;
            }
        }
    }

    private static int GetByteLength(int rows, int columns, int blockSize)
        => new Ternary2Shape(rows, columns, blockSize).ByteLength;
}

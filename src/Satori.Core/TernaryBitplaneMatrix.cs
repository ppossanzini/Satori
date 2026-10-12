using System.Buffers.Binary;

namespace Satori.Core;

/// <summary>Immutable ternary weights with separate positive and negative bitplanes.</summary>
public sealed class TernaryBitplaneMatrix
{
    private readonly byte[] _data;
    internal Ternary2Shape Shape { get; }
    internal ReadOnlySpan<byte> Data => _data;
    public int Rows => Shape.Rows;
    public int Columns => Shape.Columns;
    public int BlockSize => Shape.BlockSize;
    public int ByteLength => _data.Length;

    private TernaryBitplaneMatrix(Ternary2Shape shape, byte[] data) { Shape = shape; _data = data; }
    public void CopyDataTo(Span<byte> destination) => _data.AsSpan().CopyTo(destination);

    public static TernaryBitplaneMatrix FromTernary2(Ternary2Matrix matrix)
    {
        var shape = matrix.Shape;
        byte[] data = new byte[shape.ByteLength];
        int planeBytes = shape.BlockSize / 8;
        for (int offset = 0; offset < data.Length; offset += shape.BlockByteLength)
        {
            var source = matrix.Data.Slice(offset, shape.BlockByteLength);
            var target = data.AsSpan(offset, shape.BlockByteLength);
            source[..4].CopyTo(target);
            for (int i = 0; i < shape.BlockSize; i++)
            {
                int code = (source[4 + i / 4] >> (2 * (i % 4))) & 3;
                if (code != 0) target[4 + (code == 2 ? planeBytes : 0) + i / 8] |= (byte)(1 << (i % 8));
            }
        }
        return new(shape, data);
    }

    public Ternary2Matrix ToTernary2()
    {
        byte[] data = new byte[ByteLength];
        int planeBytes = BlockSize / 8;
        for (int offset = 0; offset < data.Length; offset += Shape.BlockByteLength)
        {
            var source = Data.Slice(offset, Shape.BlockByteLength);
            var target = data.AsSpan(offset, Shape.BlockByteLength);
            source[..4].CopyTo(target);
            for (int i = 0; i < BlockSize; i++)
            {
                int positive = (source[4 + i / 8] >> (i % 8)) & 1;
                int negative = (source[4 + planeBytes + i / 8] >> (i % 8)) & 1;
                target[4 + i / 4] |= (byte)((positive | negative << 1) << (2 * (i % 4)));
            }
        }
        return Ternary2Matrix.FromPacked(Rows, Columns, BlockSize, data);
    }

    public static TernaryBitplaneMatrix FromPacked(int rows, int columns, int blockSize, ReadOnlySpan<byte> payload)
    {
        var shape = new Ternary2Shape(rows, columns, blockSize);
        if (payload.Length != shape.ByteLength) throw new ArgumentException("Incorrect bitplane payload length.", nameof(payload));
        var owned = payload.ToArray();
        Validate(shape, owned);
        return new(shape, owned);
    }

    private static void Validate(Ternary2Shape shape, ReadOnlySpan<byte> data)
    {
        int planeBytes = shape.BlockSize / 8;
        for (int row = 0; row < shape.Rows; row++)
        for (int b = 0; b < shape.BlocksPerRow; b++)
        {
            var block = data.Slice((row * shape.BlocksPerRow + b) * shape.BlockByteLength, shape.BlockByteLength);
            float scale = BinaryPrimitives.ReadSingleLittleEndian(block);
            if (!float.IsFinite(scale) || scale < 0 || BitConverter.SingleToInt32Bits(scale) == int.MinValue)
                throw new ArgumentException("Invalid bitplane scale.", nameof(data));
            int count = Math.Min(shape.BlockSize, shape.Columns - b * shape.BlockSize);
            for (int i = 0; i < planeBytes; i++)
            {
                byte positive = block[4 + i], negative = block[4 + planeBytes + i];
                int valid = Math.Clamp(count - i * 8, 0, 8);
                int padding = ~((1 << valid) - 1);
                if ((positive & negative) != 0 || ((positive | negative) & padding) != 0 ||
                    (scale == 0 && (positive | negative) != 0))
                    throw new ArgumentException("Overlapping bits, invalid padding or noncanonical zero.", nameof(data));
            }
        }
    }

    public void Save(string path)
    {
        Span<byte> header = stackalloc byte[32];
        TernaryBitplaneFileFormat.WriteHeader(header, Shape);
        using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        file.Write(header);
        file.Write(_data);
    }

    public static TernaryBitplaneMatrix Load(string path)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (file.Length < 32) throw new InvalidDataException("Truncated bitplane header.");
        Span<byte> header = stackalloc byte[32];
        file.ReadExactly(header);
        try
        {
            var shape = TernaryBitplaneFileFormat.ReadHeader(header);
            if (file.Length != 32L + shape.ByteLength) throw new InvalidDataException("Incorrect bitplane file length.");
            var data = new byte[shape.ByteLength];
            file.ReadExactly(data);
            Validate(shape, data);
            return new(shape, data);
        }
        catch (Exception exception) when (exception is ArgumentException or OverflowException or EndOfStreamException)
        {
            throw new InvalidDataException("Invalid bitplane file.", exception);
        }
    }
}

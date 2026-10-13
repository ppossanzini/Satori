using System.Runtime.InteropServices;

namespace Satori.Core;

/// <summary>Decodes one validated quantized row without expanding the matrix.</summary>
public static class QuantizedRow
{
    public static void DecodeQ8(ReadOnlySpan<byte> bytes, Span<float> output)
    {
        if (output.Length == 0 || output.Length % 32 != 0 || bytes.Length != checked(output.Length / 32 * 34))
            throw new ArgumentException("Invalid Q8_0 row dimensions.");
        if (bytes.Overlaps(MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Row output overlaps packed input.");
        for (int block = 0; block < output.Length / 32; block++)
        {
            var data = bytes.Slice(block * 34, 34);
            float scale = QuantizedMatVec.Scale(data);
            for (int j = 0; j < 32; j++) output[block * 32 + j] = scale * unchecked((sbyte)data[2 + j]);
        }
    }

    /// <summary>
    /// Decodes a full row of <paramref name="shape"/> for any supported format. Used for embedding lookups,
    /// where the whole row is materialised once per token.
    /// </summary>
    public static void DecodeRow(ReadOnlySpan<byte> bytes, QuantizedMatrixShape shape, Span<float> output)
    {
        int blocks = shape.Columns / shape.BlockElements;
        if (output.Length != shape.Columns || bytes.Length != checked(blocks * shape.BlockBytes))
            throw new ArgumentException("Invalid row dimensions for the format.");
        if (bytes.Overlaps(MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Row output overlaps packed input.");
        for (int block = 0; block < blocks; block++)
        {
            var data = bytes.Slice(block * shape.BlockBytes, shape.BlockBytes);
            var slice = output.Slice(block * shape.BlockElements, shape.BlockElements);
            switch (shape.Format)
            {
                case BlockQuantization.Q8_0: DecodeQ8(data, slice); break;
                case BlockQuantization.Q4_K: Q4KCodec.DecodeBlock(data, slice); break;
                case BlockQuantization.Q6_K: Q6KCodec.DecodeBlock(data, slice); break;
                default: throw new NotSupportedException($"Row decode for {shape.Format} is not implemented.");
            }
        }
    }
}

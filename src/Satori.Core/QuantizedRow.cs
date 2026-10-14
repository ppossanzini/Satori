using System.Runtime.InteropServices;

namespace Satori.Core;

/// <summary>Decodes one validated quantized row without expanding the matrix.</summary>
public static class QuantizedRow
{
    /// <summary>Decodes a whole row of Q8_0 blocks, 32 weights per 34 bytes.</summary>
    public static void DecodeQ8(ReadOnlySpan<byte> bytes, Span<float> output)
    {
        if (output.Length == 0 || output.Length % 32 != 0 || bytes.Length != checked(output.Length / 32 * 34))
            throw new ArgumentException("Invalid Q8_0 row dimensions.");
        if (bytes.Overlaps(MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Row output overlaps packed input.");
        for (int block = 0; block < output.Length / 32; block++)
            DecodeQ8Block(bytes.Slice(block * 34, 34), output.Slice(block * 32, 32));
    }

    /// <summary>Decodes one Q8_0 block: an FP16 scale then 32 signed bytes.</summary>
    public static void DecodeQ8Block(ReadOnlySpan<byte> block, Span<float> output)
    {
        if (block.Length != 34 || output.Length != 32) throw new ArgumentException("Invalid Q8_0 block dimensions.");
        float scale = QuantizedMatVec.Scale(block);
        for (int j = 0; j < 32; j++) output[j] = scale * unchecked((sbyte)block[2 + j]);
    }

    /// <summary>Decodes one Q4_0 block: an FP16 scale then 32 four-bit codes biased by -8.</summary>
    public static void DecodeQ4Block(ReadOnlySpan<byte> block, Span<float> output)
    {
        if (block.Length != 18 || output.Length != 32) throw new ArgumentException("Invalid Q4_0 block dimensions.");
        float scale = QuantizedMatVec.Scale(block);
        for (int j = 0; j < 16; j++)
        {
            output[j] = scale * ((block[2 + j] & 15) - 8);
            output[j + 16] = scale * ((block[2 + j] >> 4) - 8);
        }
    }

    /// <summary>
    /// Decodes exactly one block of <paramref name="format"/>, whatever its super-block size. This is the one
    /// place that knows how many weights a block holds and which codec reconstructs it, so a caller that
    /// processes blocks one at a time never repeats the format table.
    /// </summary>
    public static void DecodeBlock(ReadOnlySpan<byte> block, BlockQuantization format, Span<float> output)
    {
        switch (format)
        {
            case BlockQuantization.Q8_0: DecodeQ8Block(block, output); break;
            case BlockQuantization.Q4_0: DecodeQ4Block(block, output); break;
            case BlockQuantization.Q4_K: Q4KCodec.DecodeBlock(block, output); break;
            case BlockQuantization.Q6_K: Q6KCodec.DecodeBlock(block, output); break;
            default: throw new NotSupportedException($"Block decode for {format} is not implemented.");
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
            DecodeBlock(bytes.Slice(block * shape.BlockBytes, shape.BlockBytes), shape.Format,
                output.Slice(block * shape.BlockElements, shape.BlockElements));
    }
}

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Satori.Core;

/// <summary>
/// GGML Q8_K activation block: 256 signed 8-bit codes, one FP32 scale and 16 group sums, in 292 bytes.
/// This is the activation-side format the integer kernels need. It is deliberately lossy: it exists so one
/// AVX2 instruction can perform 64 multiply-adds on integers instead of 8 on floats, which is the difference
/// between a projection that streams activations and one that does not.
/// </summary>
/// <remarks>
/// Layout pinned from the ggml revision in <c>docs/gguf-subset-v1.md</c> (<c>block_q8_K</c>): an FP32 scale
/// <c>d</c>, then <c>qs[256]</c>, then <c>bsums[16]</c> holding the sum of each group of 16 codes.
/// The quantizer reproduces <c>quantize_row_q8_K_ref</c>: the code scale is <c>-127/max</c> where
/// <c>max</c> is the signed element of largest magnitude, so codes are mostly positive and the unsigned
/// multiply of the integer kernels is well defined.
/// </remarks>
public static class Q8KCodec
{
    /// <summary>Weights or activations covered by one block.</summary>
    public const int ElementsPerBlock = 256;

    /// <summary>Codes in one bsums group.</summary>
    public const int GroupSize = 16;

    /// <summary>Groups per block.</summary>
    public const int Groups = ElementsPerBlock / GroupSize;

    /// <summary>Bytes occupied by one block: 4 + 256 + 16*2.</summary>
    public const int BytesPerBlock = 292;

    /// <summary>Codes start here.</summary>
    public const int QuantsOffset = 4;

    /// <summary>Group sums start here.</summary>
    public const int SumsOffset = QuantsOffset + ElementsPerBlock;

    /// <summary>
    /// Quantizes one block of <paramref name="input"/> into <paramref name="block"/> of
    /// <see cref="BytesPerBlock"/> bytes. An all-zero block gets a zero scale and zero codes, and the caller
    /// must treat its product as zero rather than reading the codes.
    /// </summary>
    public static void QuantizeBlock(ReadOnlySpan<float> input, Span<byte> block)
    {
        if (input.Length != ElementsPerBlock || block.Length != BytesPerBlock)
            throw new ArgumentException("Invalid Q8_K block dimensions.");
        float max = 0, magnitude = 0;
        for (int j = 0; j < ElementsPerBlock; j++)
        {
            float value = input[j];
            float absolute = Math.Abs(value);
            if (absolute > magnitude)
            {
                magnitude = absolute;
                max = value;
            }
        }
        if (magnitude == 0)
        {
            block.Clear();
            return;
        }
        float inverse = -127f / max;
        Span<sbyte> codes = MemoryMarshal.Cast<byte, sbyte>(block.Slice(QuantsOffset, ElementsPerBlock));
        for (int j = 0; j < ElementsPerBlock; j++)
            codes[j] = (sbyte)Math.Min(127, NearestInteger(inverse * input[j]));
        Span<short> sums = MemoryMarshal.Cast<byte, short>(block.Slice(SumsOffset, Groups * sizeof(short)));
        for (int group = 0; group < Groups; group++)
        {
            int total = 0;
            for (int j = 0; j < GroupSize; j++) total += codes[group * GroupSize + j];
            sums[group] = (short)total;
        }
        BinaryPrimitives.WriteSingleLittleEndian(block, 1f / inverse);
    }

    /// <summary>The block scale, or zero for an all-zero block.</summary>
    public static float Scale(ReadOnlySpan<byte> block) => BinaryPrimitives.ReadSingleLittleEndian(block);

    /// <summary>
    /// Quantizes a whole activation vector, one block at a time. The caller keeps the result across every
    /// row of the matrix, which is the entire point: the activation is read once per matrix, not once per row.
    /// </summary>
    public static void QuantizeRow(ReadOnlySpan<float> input, Span<byte> output)
    {
        int blocks = input.Length / ElementsPerBlock;
        if (input.Length % ElementsPerBlock != 0 || output.Length != checked(blocks * BytesPerBlock))
            throw new ArgumentException("Invalid Q8_K row dimensions.");
        for (int block = 0; block < blocks; block++)
            QuantizeBlock(input.Slice(block * ElementsPerBlock, ElementsPerBlock),
                output.Slice(block * BytesPerBlock, BytesPerBlock));
    }

    /// <summary>Bytes a row of <paramref name="columns"/> floats needs when quantized.</summary>
    public static int RowBytes(int columns)
    {
        if (columns <= 0 || columns % ElementsPerBlock != 0)
            throw new ArgumentOutOfRangeException(nameof(columns), $"Columns must be a positive multiple of {ElementsPerBlock}.");
        return checked(columns / ElementsPerBlock * BytesPerBlock);
    }

    /// <summary>The signed codes of one block, which the weight codecs multiply.</summary>
    public static ReadOnlySpan<sbyte> Codes(ReadOnlySpan<byte> block) =>
        MemoryMarshal.Cast<byte, sbyte>(block.Slice(QuantsOffset, ElementsPerBlock));

    /// <summary>The 16 group sums of one block.</summary>
    public static ReadOnlySpan<short> GroupSums(ReadOnlySpan<byte> block) =>
        MemoryMarshal.Cast<byte, short>(block.Slice(SumsOffset, Groups * sizeof(short)));

    /// <summary>True when the block carries no information, so a product with it is zero.</summary>
    public static bool IsZero(ReadOnlySpan<byte> block) => Scale(block) == 0;

    /// <summary>
    /// Rounds to the nearest integer exactly as the reference's <c>nearest_int</c> does. The reference adds
    /// <c>12582912</c> (1.5 * 2^23) in single precision and reads the low mantissa bits: the addition rounds
    /// to nearest-even, so the trick itself is the rounding step and cannot be replaced by a language round
    /// without changing which side of a tie an element falls on. It is reproduced literally for that reason.
    /// </summary>
    private static int NearestInteger(float value)
    {
        float shifted = value + 12582912f;
        int bits = BitConverter.SingleToInt32Bits(shifted);
        return (bits & 0x007fffff) - 0x00400000;
    }
}

using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Satori.Core;

/// <summary>
/// GGML Q6_K super-block codec: 256 weights in 210 bytes. Layout pinned from the ggml revision in
/// <c>docs/gguf-subset-v1.md</c> (<c>ggml-common.h</c>, <c>block_q6_K</c>): 128 lower-4-bit bytes in
/// <c>ql</c>, 64 upper-2-bit bytes in <c>qh</c>, 16 signed 8-bit sub-scales in <c>scales</c> (one per 16
/// weights), and an FP16 super-scale <c>d</c>.
/// A reconstructed weight is <c>d * scale[j] * (q - 32)</c> where <c>q</c> is a 6-bit unsigned code.
/// </summary>
public static class Q6KCodec
{
    /// <summary>Weights covered by one super-block.</summary>
    public const int ElementsPerBlock = 256;

    /// <summary>Bytes occupied by one super-block.</summary>
    public const int BytesPerBlock = 210;

    /// <summary>Dot product of one super-block's reconstructed weights with the matching input window.</summary>
    public static float DotBlock(ReadOnlySpan<byte> block, ReadOnlySpan<float> input)
    {
        if (block.Length != BytesPerBlock || input.Length != ElementsPerBlock)
            throw new ArgumentException("Q6_K block or input window has the wrong length.");
        float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[208..]));
        var ql = block[..128];
        var qh = block.Slice(128, 64);
        var scales = MemoryMarshal.Cast<byte, sbyte>(block.Slice(192, 16));
        float total = 0;
        for (int half = 0; half < 2; half++)
        {
            var lql = ql.Slice(half * 64, 64);
            var lqh = qh.Slice(half * 32, 32);
            var lsc = scales.Slice(half * 8, 8);
            int baseIndex = half * 128;
            for (int l = 0; l < 32; l++)
            {
                int sub = l / 16;
                int q1 = ((lql[l] & 0xF) | (((lqh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((lql[l + 32] & 0xF) | (((lqh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((lql[l] >> 4) | (((lqh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((lql[l + 32] >> 4) | (((lqh[l] >> 6) & 3) << 4)) - 32;
                total += d * lsc[sub + 0] * q1 * input[baseIndex + l + 0];
                total += d * lsc[sub + 2] * q2 * input[baseIndex + l + 32];
                total += d * lsc[sub + 4] * q3 * input[baseIndex + l + 64];
                total += d * lsc[sub + 6] * q4 * input[baseIndex + l + 96];
            }
        }
        return total;
    }

    /// <summary>Reconstructs the 256 weights of one super-block into <paramref name="weights"/>.</summary>
    public static void DecodeBlock(ReadOnlySpan<byte> block, Span<float> weights)
    {
        if (block.Length != BytesPerBlock || weights.Length != ElementsPerBlock)
            throw new ArgumentException("Q6_K block or output window has the wrong length.");
        float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[208..]));
        var ql = block[..128];
        var qh = block.Slice(128, 64);
        var scales = MemoryMarshal.Cast<byte, sbyte>(block.Slice(192, 16));
        for (int half = 0; half < 2; half++)
        {
            var lql = ql.Slice(half * 64, 64);
            var lqh = qh.Slice(half * 32, 32);
            var lsc = scales.Slice(half * 8, 8);
            int baseIndex = half * 128;
            for (int l = 0; l < 32; l++)
            {
                int sub = l / 16;
                int q1 = ((lql[l] & 0xF) | (((lqh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((lql[l + 32] & 0xF) | (((lqh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((lql[l] >> 4) | (((lqh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((lql[l + 32] >> 4) | (((lqh[l] >> 6) & 3) << 4)) - 32;
                weights[baseIndex + l + 0] = d * lsc[sub + 0] * q1;
                weights[baseIndex + l + 32] = d * lsc[sub + 2] * q2;
                weights[baseIndex + l + 64] = d * lsc[sub + 4] * q3;
                weights[baseIndex + l + 96] = d * lsc[sub + 6] * q4;
            }
        }
    }

    /// <summary>Checks that the super-block scale is finite.</summary>
    public static bool AreScalesFinite(ReadOnlySpan<byte> block) =>
        float.IsFinite((float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[208..])));

    /// <summary>
    /// One super-block against one Q8_K activation block: the integer form of <see cref="DotBlock"/>, where the
    /// floating-point input window is replaced by the signed codes the reference quantizer produced.
    /// </summary>
    /// <remarks>
    /// The sub-scale selection is written out deliberately instead of vectorised along the lines of the
    /// reference kernel. The reference multiplies <b>uncentred</b> six-bit codes there and removes the
    /// <c>-32</c> offset afterwards through the activation's group sums, which is a different grouping from
    /// the one the format's own decoder uses; copying it without deriving it once produced a sign error in an
    /// AVX2 attempt. This form is the decoder's own arithmetic with the input replaced, so it cannot describe
    /// a different weight layout from the decoder that the independent fixture already validates.
    /// </remarks>
    public static float DotBlockQuantized(ReadOnlySpan<byte> block, ReadOnlySpan<byte> activation)
    {
        if (block.Length != BytesPerBlock || activation.Length != Q8KCodec.BytesPerBlock)
            throw new ArgumentException("Q6_K block or Q8_K activation has the wrong length.");
        if (Q8KCodec.IsZero(activation)) return 0;
        float d = Q8KCodec.Scale(activation) *
            (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[208..]));
        return d * DotProduct(block, activation);
    }

    /// <summary>
    /// <c>Σ scale_s * Σ (q - 32) * code</c> over the sixteen 16-weight sub-blocks, in exact integer
    /// arithmetic. Kept separate from <see cref="DotBlockQuantized"/> so a vectorised form can be compared
    /// against the integer quantity directly, where any difference is a defect and not a rounding artefact.
    /// </summary>
    public static int DotProduct(ReadOnlySpan<byte> block, ReadOnlySpan<byte> activation)
    {
        if (block.Length != BytesPerBlock || activation.Length != Q8KCodec.BytesPerBlock)
            throw new ArgumentException("Q6_K block or Q8_K activation has the wrong length.");
        var codes = Q8KCodec.Codes(activation);
        var ql = block[..128];
        var qh = block.Slice(128, 64);
        var scales = MemoryMarshal.Cast<byte, sbyte>(block.Slice(192, 16));
        int total = 0;
        for (int half = 0; half < 2; half++)
        {
            var lql = ql.Slice(half * 64, 64);
            var lqh = qh.Slice(half * 32, 32);
            var lsc = scales.Slice(half * 8, 8);
            int baseIndex = half * 128;
            // The four codes of one l pair the low and high nibbles of two ql bytes with the four 2-bit fields
            // of one qh byte, and each lands in its own quarter of the half-block under its own scale. The
            // index expressions below are the decoder's, with codes in place of the input window.
            for (int l = 0; l < 32; l++)
            {
                int sub = l / 16;
                int q1 = ((lql[l] & 0xF) | (((lqh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((lql[l + 32] & 0xF) | (((lqh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((lql[l] >> 4) | (((lqh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((lql[l + 32] >> 4) | (((lqh[l] >> 6) & 3) << 4)) - 32;
                total += lsc[sub + 0] * q1 * codes[baseIndex + l + 0];
                total += lsc[sub + 2] * q2 * codes[baseIndex + l + 32];
                total += lsc[sub + 4] * q3 * codes[baseIndex + l + 64];
                total += lsc[sub + 6] * q4 * codes[baseIndex + l + 96];
            }
        }
        return total;
    }
}

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
}

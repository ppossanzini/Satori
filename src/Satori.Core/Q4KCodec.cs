using System.Buffers.Binary;

namespace Satori.Core;

/// <summary>
/// GGML Q4_K super-block codec: 256 weights in 144 bytes. Layout pinned from the ggml revision in
/// <c>docs/gguf-subset-v1.md</c> (<c>ggml-common.h</c>, <c>block_q4_K</c>): an FP16 super-scale
/// <c>d</c>, an FP16 super-min scale <c>dmin</c>, eight 6-bit scales and eight 6-bit mins packed into
/// <c>scales[12]</c>, and 256 four-bit quants in <c>qs[128]</c>.
/// A reconstructed weight is <c>d * scale[j] * q - dmin * min[j]</c>.
/// </summary>
public static class Q4KCodec
{
    /// <summary>Weights covered by one super-block.</summary>
    public const int ElementsPerBlock = 256;

    /// <summary>Bytes occupied by one super-block.</summary>
    public const int BytesPerBlock = 144;

    /// <summary>Number of 32-weight sub-blocks per super-block, each with its own 6-bit scale and min.</summary>
    private const int SubBlocks = 8;

    /// <summary>Extracts the 6-bit scale and 6-bit min of sub-block <paramref name="j"/> (0..7).</summary>
    public static void GetScaleMin(int j, ReadOnlySpan<byte> scales, out int scale, out int min)
    {
        if (j < 4)
        {
            scale = scales[j] & 63;
            min = scales[j + 4] & 63;
        }
        else
        {
            scale = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
            min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        }
    }

    /// <summary>Dot product of one super-block's reconstructed weights with the matching input window.</summary>
    public static float DotBlock(ReadOnlySpan<byte> block, ReadOnlySpan<float> input)
    {
        if (block.Length != BytesPerBlock || input.Length != ElementsPerBlock)
            throw new ArgumentException("Q4_K block or input window has the wrong length.");
        float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
        float dmin = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[2..]));
        var scales = block.Slice(4, 12);
        var quants = block.Slice(16, 128);
        float total = 0;
        for (int group = 0; group < 4; group++)
        {
            var qs = quants.Slice(group * 32, 32);
            var low = input.Slice(group * 64, 32);
            var high = input.Slice(group * 64 + 32, 32);
            GetScaleMin(group * 2, scales, out int s0, out int n0);
            GetScaleMin(group * 2 + 1, scales, out int s1, out int n1);
            float dl = d * s0, ml = dmin * n0, dh = d * s1, mh = dmin * n1;
            float accLow = 0, accHigh = 0;
            for (int l = 0; l < 32; l++)
            {
                accLow += (dl * (qs[l] & 0xF) - ml) * low[l];
                accHigh += (dh * (qs[l] >> 4) - mh) * high[l];
            }
            total += accLow + accHigh;
        }
        return total;
    }

    /// <summary>Reconstructs the 256 weights of one super-block into <paramref name="weights"/>.</summary>
    public static void DecodeBlock(ReadOnlySpan<byte> block, Span<float> weights)
    {
        if (block.Length != BytesPerBlock || weights.Length != ElementsPerBlock)
            throw new ArgumentException("Q4_K block or output window has the wrong length.");
        float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
        float dmin = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[2..]));
        var scales = block.Slice(4, 12);
        var quants = block.Slice(16, 128);
        for (int group = 0; group < 4; group++)
        {
            var qs = quants.Slice(group * 32, 32);
            GetScaleMin(group * 2, scales, out int s0, out int n0);
            GetScaleMin(group * 2 + 1, scales, out int s1, out int n1);
            float dl = d * s0, ml = dmin * n0, dh = d * s1, mh = dmin * n1;
            for (int l = 0; l < 32; l++)
            {
                weights[group * 64 + l] = dl * (qs[l] & 0xF) - ml;
                weights[group * 64 + 32 + l] = dh * (qs[l] >> 4) - mh;
            }
        }
    }

    /// <summary>Checks that both super-block scales are finite.</summary>
    public static bool AreScalesFinite(ReadOnlySpan<byte> block) =>
        float.IsFinite((float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block))) &&
        float.IsFinite((float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block[2..])));
}

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

/// <summary>
/// AVX2 kernels for the K super-block formats (Q4_K, Q6_K). They mirror <see cref="Q4KCodec"/> and
/// <see cref="Q6KCodec"/> exactly, so a matching result against the scalar path is pure parity. Only reached
/// when AVX2 is available; the scalar kernels remain the reference and the fallback.
/// </summary>
internal static class QuantizedKAvx2Kernel
{
    internal static void Execute(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        if (shape.Format == BlockQuantization.Q4_K) ExecuteQ4K(data, shape, input, output);
        else ExecuteQ6K(data, shape, input, output);
    }

    /// <summary>Positions one worker accumulates at a time; the accumulator is one float per position.</summary>
    internal const int TilePositions = 256;

    /// <summary>
    /// Batched integer product for the K-quantized formats. <paramref name="activations"/> holds
    /// <paramref name="positions"/> Q8_K rows, quantized once for the whole matrix, so every row multiplies
    /// the same integer activation.
    /// </summary>
    /// <remarks>
    /// The loop is row, then block, then position: a row's weight blocks are streamed once and stay in the
    /// first-level cache, and the activation rows are swept once per weight row. The block constants that do not
    /// depend on the position are read once per block rather than once per position, which was measured to
    /// matter more than the products themselves: for Q4_K the two block scales and the eight 6-bit scales and
    /// mins, for Q6_K the single block scale.
    /// <para>
    /// Both K formats get a vectorised integer product, and both are required to return the scalar codec's
    /// integer exactly. Q4_K's eight sub-blocks of thirty-two weights each land one sub-scale per
    /// thirty-two-weight step; Q6_K's sixteen sub-blocks of sixteen weights land two, so its second
    /// multiply-add consumes a lane pattern with the two scales interleaved by half. Neither kernel copies the
    /// reference implementation's formulation of the centring term: the reference removes it through a scale
    /// grouping the format's own decoder does not share, and reproducing that grouping without deriving it once
    /// produced a sign error. Both derive it from the decoder's index mapping instead.
    /// </para>
    /// </remarks>
    internal static unsafe void ExecuteQuantizedTiled(byte* weights, QuantizedMatrixShape shape,
        byte* activations, int positions, float* output, int begin, int end)
    {
        if (positions > TilePositions)
            throw new ArgumentOutOfRangeException(nameof(positions), $"At most {TilePositions} positions per call.");
        int blocksPerRow = shape.Columns / shape.BlockElements;
        int activationRowBytes = Q8KCodec.RowBytes(shape.Columns);
        int blockBytes = shape.BlockBytes;
        bool q4 = shape.Format == BlockQuantization.Q4_K;
        var lowMask = Vector256.Create((byte)15);
        Span<int> subScales = stackalloc int[8];
        Span<int> subMins = stackalloc int[8];
        Span<float> accumulator = stackalloc float[TilePositions];
        accumulator = accumulator[..positions];
        for (int row = begin; row < end; row++)
        {
            accumulator.Clear();
            byte* weightRow = weights + (long)row * blocksPerRow * blockBytes;
            for (int block = 0; block < blocksPerRow; block++)
            {
                byte* blockWeights = weightRow + block * blockBytes;
                float d = 0, dmin = 0;
                if (q4)
                {
                    byte* scales = blockWeights + 4;
                    var packedScales = new ReadOnlySpan<byte>(scales, 12);
                    for (int sub = 0; sub < 8; sub++)
                        Q4KCodec.GetScaleMin(sub, packedScales, out subScales[sub], out subMins[sub]);
                    d = Half(new ReadOnlySpan<byte>(blockWeights, 2));
                    dmin = Half(new ReadOnlySpan<byte>(blockWeights + 2, 2));
                }
                else
                {
                    d = Half(new ReadOnlySpan<byte>(blockWeights + Q6KCodec.BytesPerBlock - 2, 2));
                }
                for (int position = 0; position < positions; position++)
                {
                    byte* activation = activations + (long)position * activationRowBytes + (long)block * Q8KCodec.BytesPerBlock;
                    var activationBlock = new ReadOnlySpan<byte>(activation, Q8KCodec.BytesPerBlock);
                    if (Q8KCodec.IsZero(activationBlock)) continue;
                    float activationScale = Q8KCodec.Scale(activationBlock);
                    // The two paths are written so that the same integer product reaches the same float: the
                    // activation scale multiplies the block scale first and the product after, in both.
                    accumulator[position] += q4
                        ? DotQuantizedQ4K(blockWeights, activation, lowMask, subScales, subMins,
                            d * activationScale, dmin * activationScale)
                        : activationScale * d * DotProductQ6K(blockWeights, activation);
                }
            }
            for (int position = 0; position < positions; position++)
                output[(long)position * shape.Rows + row] = accumulator[position];
        }
    }


    /// <summary>
    /// One Q4_K weight block against one Q8_K activation block. The nibbles are unsigned and the codes are
    /// signed, so one unsigned-times-signed multiply-add handles 64 products, and the 6-bit min of each
    /// sub-block multiplies the code sum the activation block already carries. <paramref name="scales"/> and
    /// <paramref name="mins"/> arrive pre-extracted because they do not depend on the position.
    /// </summary>
    private static unsafe float DotQuantizedQ4K(byte* weights, byte* activation, Vector256<byte> lowMask,
        ReadOnlySpan<int> scales, ReadOnlySpan<int> mins, float d, float dmin)
    {
        byte* quants = weights + 16;
        byte* codes = activation + Q8KCodec.QuantsOffset;
        short* sums = (short*)(activation + Q8KCodec.SumsOffset);
        var accumulator = Vector256<int>.Zero;
        int minTotal = 0;
        for (int sub = 0; sub < 8; sub++)
        {
            var packed = Avx.LoadVector256(quants + sub / 2 * 32);
            var nibbles = (sub & 1) == 0
                ? Avx2.And(packed, lowMask)
                : Avx2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), lowMask);
            var groupCodes = Avx.LoadVector256(codes + sub * 32).AsSByte();
            var pairs = Avx2.MultiplyAddAdjacent(nibbles, groupCodes);
            accumulator = Avx2.Add(accumulator, Avx2.MultiplyAddAdjacent(pairs, Vector256.Create((short)scales[sub])));
            minTotal += mins[sub] * (sums[sub * 2] + sums[sub * 2 + 1]);
        }
        return d * HorizontalSum(accumulator) - dmin * minTotal;
    }

    private static int HorizontalSum(Vector256<int> value)
    {
        var halves = Sse2.Add(value.GetLower(), value.GetUpper());
        halves = Sse2.Add(halves, Sse2.Shuffle(halves, 0x4e));
        halves = Sse2.Add(halves, Sse2.Shuffle(halves, 0xb1));
        return halves.ToScalar();
    }

    /// <summary>
    /// <c>Σ scale_s * Σ (q - 32) * code</c> over the sixteen 16-weight sub-blocks, in exact integer
    /// arithmetic, vectorised. This is the same quantity <see cref="Q6KCodec.DotProduct"/> computes scalars
    /// for, and the two are required to agree exactly: the products, the sub-scale weighting and the centring
    /// term are all integers, so a difference is a defect and not a rounding artefact.
    /// </summary>
    /// <remarks>
    /// The block is walked as eight groups of thirty-two weights, which is the width the unsigned-times-signed
    /// byte multiply-add works at, and is the same walk <see cref="DotQuantizedQ4K"/> makes. Within a group
    /// the thirty-two six-bit quants are assembled from one thirty-two-byte <c>ql</c> word and one
    /// thirty-two-byte <c>qh</c> word: the low nibble of <c>ql</c> and one two-bit field of <c>qh</c> in the
    /// first two groups, the high nibble in the last two, and the second <c>ql</c> word of the half for the
    /// odd groups. The index mapping is the decoder's, which the independent fixture already validates; it is
    /// reproduced here rather than taken from the reference kernel's formulation, because the reference
    /// multiplies <b>uncentred</b> codes and removes the offset afterwards through a scale grouping this
    /// format's decoder does not share, and copying that grouping without deriving it once produced a sign
    /// error.
    /// <para>
    /// The two differences from the Q4_K walk follow from the format. A Q6_K sub-block holds sixteen weights
    /// and not thirty-two, so each thirty-two-weight step carries two sub-scales rather than one, laid out as
    /// the sixteen lanes the second multiply-add consumes. The code offset is a constant thirty-two, so the
    /// term that removes it is <c>32 * Σ scale_s * Σ code</c>, and the inner group sums are the ones the Q8_K
    /// block already carries, which is why the centring costs one integer per sub-block.
    /// </para>
    /// </remarks>
    private static unsafe int DotProductQ6K(byte* block, byte* activation)
    {
        byte* ql = block;
        byte* qh = block + 128;
        sbyte* scales = (sbyte*)(block + 192);
        byte* codes = activation + Q8KCodec.QuantsOffset;
        short* sums = (short*)(activation + Q8KCodec.SumsOffset);
        var lowNibble = Vector256.Create((byte)0x0F);
        var twoBits = Vector256.Create((byte)0x03);
        var upperHalf = Vector256.Create((byte)0x30);
        var accumulator = Vector256<int>.Zero;
        int centring = 0;
        for (int group = 0; group < 8; group++)
        {
            int half = group / 4;
            int quarter = group % 4;
            byte* qlWord = ql + half * 64 + (quarter % 2) * 32;
            byte* qhWord = qh + half * 32;
            var packed = Avx.LoadVector256(qlWord);
            var nibbles = quarter >= 2
                ? Avx2.And(Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(), lowNibble)
                : Avx2.And(packed, lowNibble);
            // A right shift of a sixteen-bit lane moves the neighbouring byte's low nibble into the byte being
            // read, and masking to two bits discards it, so each byte yields its own field. The shift count is
            // written as four constants rather than one variable: the immediate form is the only one the
            // sixteen-bit shift has, and a variable count would make the compiler materialise it elsewhere.
            var high = Avx2.And(Q6KHighField(Avx.LoadVector256(qhWord), quarter), twoBits);
            // The same trick on the way back: shifting the sixteen-bit lane left moves the field to bits four
            // and five of its own byte, and the mask keeps exactly those two bits.
            var quants = Avx2.Or(nibbles,
                Avx2.And(Avx2.ShiftLeftLogical(high.AsUInt16(), 4).AsByte(), upperHalf));
            var groupCodes = Avx.LoadVector256(codes + group * 32).AsSByte();
            var pairs = Avx2.MultiplyAddAdjacent(quants, groupCodes);
            sbyte low = scales[group * 2];
            sbyte highScale = scales[group * 2 + 1];
            accumulator = Avx2.Add(accumulator, Avx2.MultiplyAddAdjacent(pairs, LaneScales(low, highScale)));
            centring += 32 * (low * sums[group * 2] + highScale * sums[group * 2 + 1]);
        }
        return HorizontalSum(accumulator) - centring;
    }

    /// <summary>
    /// The two-bit field of <paramref name="packed"/> that <paramref name="quarter"/> selects, moved down to
    /// the low bits of each byte. Written as constants because a variable shift count is not the immediate form.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Q6KHighField(Vector256<byte> packed, int quarter) => quarter switch
    {
        0 => packed,
        1 => Avx2.ShiftRightLogical(packed.AsUInt16(), 2).AsByte(),
        2 => Avx2.ShiftRightLogical(packed.AsUInt16(), 4).AsByte(),
        _ => Avx2.ShiftRightLogical(packed.AsUInt16(), 6).AsByte(),
    };

    /// <summary>
    /// The two sub-scales of one thirty-two-weight step, laid out for the second multiply-add: it pairs adjacent
    /// sixteen-bit lanes, so the first eight lanes weight the first sixteen weights and the last eight the
    /// second sixteen.
    /// </summary>
    private static Vector256<short> LaneScales(sbyte low, sbyte high) =>
        Vector256.Create(Vector128.Create((short)low), Vector128.Create((short)high));

    private static void ExecuteQ4K(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        ref float source = ref MemoryMarshal.GetReference(input);
        var lowMask = Vector256.Create(15);
        int blocksPerRow = shape.Columns / Q4KCodec.ElementsPerBlock;
        Span<int> scaleMin = stackalloc int[4];
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int b = 0; b < blocksPerRow; b++)
            {
                var block = data.Slice((row * blocksPerRow + b) * Q4KCodec.BytesPerBlock, Q4KCodec.BytesPerBlock);
                float d = Half(block);
                float dmin = Half(block[2..]);
                var scales = block.Slice(4, 12);
                var qs = block.Slice(16, 128);
                int columnBase = b * Q4KCodec.ElementsPerBlock;
                for (int g = 0; g < 4; g++)
                {
                    var q = qs.Slice(g * 32, 32);
                    Q4KCodec.GetScaleMin(g * 2, scales, out scaleMin[0], out scaleMin[2]);
                    Q4KCodec.GetScaleMin(g * 2 + 1, scales, out scaleMin[1], out scaleMin[3]);
                    var lowScale = Vector256.Create(d * scaleMin[0]);
                    var lowBias = Vector256.Create(dmin * scaleMin[2]);
                    var highScale = Vector256.Create(d * scaleMin[1]);
                    var highBias = Vector256.Create(dmin * scaleMin[3]);
                    var accumulator = Vector256<float>.Zero;
                    for (int j = 0; j < 32; j += 8)
                    {
                        var packed = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(q[j..])).AsByte();
                        var codes = Avx2.ConvertToVector256Int32(packed);
                        var low = Avx.ConvertToVector256Single(Avx2.And(codes, lowMask));
                        var high = Avx.ConvertToVector256Single(Avx2.ShiftRightLogical(codes, 4));
                        var lowWeights = Avx.Subtract(Avx.Multiply(low, lowScale), lowBias);
                        var highWeights = Avx.Subtract(Avx.Multiply(high, highScale), highBias);
                        accumulator = Avx.Add(accumulator, Avx.Multiply(lowWeights, Vector256.LoadUnsafe(ref source, (nuint)(columnBase + g * 64 + j))));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(highWeights, Vector256.LoadUnsafe(ref source, (nuint)(columnBase + g * 64 + 32 + j))));
                    }
                    total += HorizontalSum(accumulator);
                }
            }
            output[row] = total;
        }
    }

    private static void ExecuteQ6K(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        ref float source = ref MemoryMarshal.GetReference(input);
        var lowMask = Vector256.Create(15);
        var three = Vector256.Create(3);
        var bias32 = Vector256.Create(32);
        int blocksPerRow = shape.Columns / Q6KCodec.ElementsPerBlock;
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int b = 0; b < blocksPerRow; b++)
            {
                var block = data.Slice((row * blocksPerRow + b) * Q6KCodec.BytesPerBlock, Q6KCodec.BytesPerBlock);
                float d = Half(block[208..]);
                var ql = block[..128];
                var qh = block.Slice(128, 64);
                var scales = MemoryMarshal.Cast<byte, sbyte>(block.Slice(192, 16));
                int columnBase = b * Q6KCodec.ElementsPerBlock;
                for (int half = 0; half < 2; half++)
                {
                    var lql = ql.Slice(half * 64, 64);
                    var lqh = qh.Slice(half * 32, 32);
                    var lsc = scales.Slice(half * 8, 8);
                    int baseIndex = columnBase + half * 128;
                    for (int l = 0; l < 32; l += 8)
                    {
                        int sub = l / 16;
                        var a1 = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(lql[l..])).AsByte();
                        var a2 = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(lql[(l + 32)..])).AsByte();
                        var bq = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(lqh[l..])).AsByte();
                        var a1i = Avx2.ConvertToVector256Int32(a1);
                        var a2i = Avx2.ConvertToVector256Int32(a2);
                        var bx = Avx2.ConvertToVector256Int32(bq);
                        var qh0 = Avx2.And(bx, three);
                        var qh1 = Avx2.And(Avx2.ShiftRightLogical(bx, (byte)2), three);
                        var qh2 = Avx2.And(Avx2.ShiftRightLogical(bx, (byte)4), three);
                        var qh3 = Avx2.And(Avx2.ShiftRightLogical(bx, (byte)6), three);
                        var c1 = Build(a1i, 0, qh0, lowMask, bias32);
                        var c2 = Build(a2i, 0, qh1, lowMask, bias32);
                        var c3 = Build(a1i, 4, qh2, lowMask, bias32);
                        var c4 = Build(a2i, 4, qh3, lowMask, bias32);
                        var w1 = Avx.Multiply(c1, Vector256.Create(d * lsc[sub + 0]));
                        var w2 = Avx.Multiply(c2, Vector256.Create(d * lsc[sub + 2]));
                        var w3 = Avx.Multiply(c3, Vector256.Create(d * lsc[sub + 4]));
                        var w4 = Avx.Multiply(c4, Vector256.Create(d * lsc[sub + 6]));
                        var accumulator = Vector256<float>.Zero;
                        accumulator = Avx.Add(accumulator, Avx.Multiply(w1, Vector256.LoadUnsafe(ref source, (nuint)(baseIndex + l))));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(w2, Vector256.LoadUnsafe(ref source, (nuint)(baseIndex + l + 32))));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(w3, Vector256.LoadUnsafe(ref source, (nuint)(baseIndex + l + 64))));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(w4, Vector256.LoadUnsafe(ref source, (nuint)(baseIndex + l + 96))));
                        total += HorizontalSum(accumulator);
                    }
                }
            }
            output[row] = total;
        }
    }

    /// <summary>Builds <c>(q - 32)</c> as float from the low nibble of <paramref name="qlV"/> and the 2-bit
    /// high field <paramref name="qhField"/>: <c>q = (qhField &lt;&lt; 4) | nibble</c>. The high nibble is used
    /// when <paramref name="nibble"/> is non-zero.</summary>
    private static Vector256<float> Build(Vector256<int> qlV, int nibble, Vector256<int> qhField,
        Vector256<int> lowMask, Vector256<int> bias32)
    {
        var ql = nibble == 0 ? Avx2.And(qlV, lowMask) : Avx2.ShiftRightLogical(qlV, (byte)4);
        var code = Avx2.Or(Avx2.ShiftLeftLogical(qhField, (byte)4), ql);
        return Avx.ConvertToVector256Single(Avx2.Subtract(code, bias32));
    }

    internal static float HorizontalSum(Vector256<float> value)
    {
        var halves = Sse.Add(value.GetLower(), value.GetUpper());
        halves = Sse.Add(halves, Sse.Shuffle(halves, halves, 0x4e));
        halves = Sse.AddScalar(halves, Sse.Shuffle(halves, halves, 0xb1));
        return halves.ToScalar();
    }

    private static float Half(ReadOnlySpan<byte> block) =>
        (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
}

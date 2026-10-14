using System.Buffers.Binary;
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

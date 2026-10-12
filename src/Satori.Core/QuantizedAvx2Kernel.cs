using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

internal static class QuantizedAvx2Kernel
{
    internal static void Execute(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        ref float source = ref MemoryMarshal.GetReference(input);
        var nibbleMask = Vector256.Create(15);
        var bias = Vector256.Create(8);
        int blocksPerRow = shape.Columns / 32;
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int b = 0; b < blocksPerRow; b++)
            {
                var bytes = data.Slice((row * blocksPerRow + b) * shape.BlockBytes, shape.BlockBytes);
                var accumulator = Vector256<float>.Zero;
                if (shape.Format == BlockQuantization.Q8_0)
                {
                    for (int j = 0; j < 32; j += 8)
                    {
                        var packed = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(bytes[(2 + j)..])).AsSByte();
                        var codes = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(packed));
                        var values = Vector256.LoadUnsafe(ref source, (nuint)(b * 32 + j));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(codes, values));
                    }
                }
                else
                {
                    for (int j = 0; j < 16; j += 8)
                    {
                        var packed = Vector128.CreateScalar(BinaryPrimitives.ReadUInt64LittleEndian(bytes[(2 + j)..])).AsByte();
                        var codes = Avx2.ConvertToVector256Int32(packed);
                        var low = Avx.ConvertToVector256Single(Avx2.Subtract(Avx2.And(codes, nibbleMask), bias));
                        var high = Avx.ConvertToVector256Single(Avx2.Subtract(Avx2.ShiftRightLogical(codes, 4), bias));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(low, Vector256.LoadUnsafe(ref source, (nuint)(b * 32 + j))));
                        accumulator = Avx.Add(accumulator, Avx.Multiply(high, Vector256.LoadUnsafe(ref source, (nuint)(b * 32 + j + 16))));
                    }
                }
                var halves = Sse.Add(accumulator.GetLower(), accumulator.GetUpper());
                halves = Sse.Add(halves, Sse.Shuffle(halves, halves, 0x4e));
                halves = Sse.AddScalar(halves, Sse.Shuffle(halves, halves, 0xb1));
                total += QuantizedMatVec.Scale(bytes) * halves.ToScalar();
            }
            output[row] = total;
        }
    }
}

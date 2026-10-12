using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

internal static class TernaryBitplaneAvx2Kernel
{
    internal static void Execute(TernaryBitplaneMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        var shape = matrix.Shape;
        int planeBytes = shape.BlockSize / 8;
        ref float source = ref MemoryMarshal.GetReference(input);
        var shifts = Vector256.Create(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u);
        var activeMask = Vector256.Create(0x101u);
        var negativeMask = Vector256.Create(0x100u);
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int b = 0; b < shape.BlocksPerRow; b++)
            {
                var bytes = matrix.Data.Slice((row * shape.BlocksPerRow + b) * shape.BlockByteLength, shape.BlockByteLength);
                int column = b * shape.BlockSize;
                int count = Math.Min(shape.BlockSize, shape.Columns - column);
                var accumulator = Vector256<float>.Zero;
                int i = 0;
                for (; i <= count - 8; i += 8)
                {
                    uint packed = (uint)(bytes[4 + i / 8] | bytes[4 + planeBytes + i / 8] << 8);
                    var lanes = Avx2.ShiftRightLogicalVariable(Vector256.Create(packed), shifts);
                    var zero = Avx2.CompareEqual(Avx2.And(lanes, activeMask), Vector256<uint>.Zero);
                    // Negative-plane bit 8 becomes the sign bit of each FP32 activation.
                    var sign = Avx2.ShiftLeftLogical(Avx2.And(lanes, negativeMask), 23);
                    var values = Vector256.LoadUnsafe(ref source, (nuint)(column + i));
                    accumulator = Avx.Add(accumulator, Avx.AndNot(zero.AsSingle(), Avx.Xor(values, sign.AsSingle())));
                }
                var halves = Sse.Add(accumulator.GetLower(), accumulator.GetUpper());
                halves = Sse.Add(halves, Sse.Shuffle(halves, halves, 0x4e));
                halves = Sse.AddScalar(halves, Sse.Shuffle(halves, halves, 0xb1));
                float partial = halves.ToScalar();
                for (; i < count; i++)
                {
                    int mask = 1 << (i % 8);
                    if ((bytes[4 + i / 8] & mask) != 0) partial += input[column + i];
                    else if ((bytes[4 + planeBytes + i / 8] & mask) != 0) partial -= input[column + i];
                }
                total += BinaryPrimitives.ReadSingleLittleEndian(bytes) * partial;
            }
            output[row] = total;
        }
    }
}

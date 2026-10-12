using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

internal static class Ternary2Avx2Kernel
{
    internal static void Execute(ReadOnlySpan<byte> data, Ternary2Shape matrix, ReadOnlySpan<float> input, Span<float> output)
    {
        int blocks = matrix.BlocksPerRow;
        int stride = matrix.BlockByteLength;
        var shifts = Vector256.Create(0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u);
        var mask = Vector256.Create(3u);
        var negativeCode = Vector256.Create(2);
        var signBit = Vector256.Create(int.MinValue);
        ref float inputStart = ref MemoryMarshal.GetReference(input);

        for (int row = 0; row < matrix.Rows; row++)
        {
            float sum = 0;
            for (int block = 0; block < blocks; block++)
            {
                var bytes = data.Slice((row * blocks + block) * stride, stride);
                float scale = BinaryPrimitives.ReadSingleLittleEndian(bytes);
                int column = block * matrix.BlockSize;
                int count = Math.Min(matrix.BlockSize, matrix.Columns - column);
                var accumulator = Vector256<float>.Zero;
                int i = 0;
                for (; i <= count - 8; i += 8)
                {
                    // Each lane extracts one two-bit code from the same eight-weight word.
                    uint packed = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(4 + i / 4));
                    var codes = Avx2.And(Avx2.ShiftRightLogicalVariable(Vector256.Create(packed), shifts), mask).AsInt32();
                    var zero = Avx2.CompareEqual(codes, Vector256<int>.Zero);
                    var negative = Avx2.And(Avx2.CompareEqual(codes, negativeCode), signBit);
                    var activations = Vector256.LoadUnsafe(ref inputStart, (nuint)(column + i));
                    var signed = Avx.Xor(activations, negative.AsSingle());
                    accumulator = Avx.Add(accumulator, Avx.AndNot(zero.AsSingle(), signed));
                }
                var halves = Sse.Add(accumulator.GetLower(), accumulator.GetUpper());
                halves = Sse.Add(halves, Sse.Shuffle(halves, halves, 0x4E));
                halves = Sse.AddScalar(halves, Sse.Shuffle(halves, halves, 0xB1));
                float partial = halves.ToScalar();
                for (; i < count; i++)
                {
                    int code = (bytes[4 + i / 4] >> (2 * (i % 4))) & 3;
                    if (code == 1) partial += input[column + i];
                    else if (code == 2) partial -= input[column + i];
                }
                sum += partial * scale;
            }
            output[row] = sum;
        }
    }
}

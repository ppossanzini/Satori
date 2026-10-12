using System.Buffers.Binary;
using Satori.Core;
using Satori.Validation;

namespace Satori.Numerics.Tests;

public sealed class TernaryBitplaneTests
{
    [Fact]
    public void IndependentGoldenAndCodecRejectInvalidPlanes()
    {
        byte[] payload = new byte[20];
        BinaryPrimitives.WriteSingleLittleEndian(payload, 0.5f);
        payload[4] = 0x89;
        payload[12] = 0x14;
        var matrix = TernaryBitplaneMatrix.FromPacked(1, 8, 64, payload);
        float[] output = new float[1];
        TernaryBitplaneMatVec.ExecuteScalar(matrix, Enumerable.Repeat(1f, 8).ToArray(), output);
        Assert.Equal(0.5f, output[0]);
        byte[] ternary = new byte[20];
        matrix.ToTernary2().CopyDataTo(ternary);
        Assert.Equal(0x61, ternary[4]); Assert.Equal(0x42, ternary[5]);
        foreach (int defect in new[] { 0, 1, 2, 3, 4 })
        {
            byte[] bad = (byte[])payload.Clone();
            switch (defect)
            {
                case 0: bad[12] |= 1; break;
                case 1: bad[5] = 1; break;
                case 2: BinaryPrimitives.WriteSingleLittleEndian(bad, -0f); break;
                case 3: BinaryPrimitives.WriteSingleLittleEndian(bad, float.NaN); break;
                case 4: BinaryPrimitives.WriteSingleLittleEndian(bad, 0f); break;
            }
            Assert.Throws<ArgumentException>(() => TernaryBitplaneMatrix.FromPacked(1, 8, 64, bad));
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void CorpusRoundTripPreservesEveryByteAndScalarResult(int blockSize)
    {
        foreach (var sample in NumericalCorpus.CriticalCases(blockSize))
        {
            var original = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns, blockSize);
            var converted = TernaryBitplaneMatrix.FromTernary2(original);
            byte[] before = new byte[original.ByteLength], after = new byte[original.ByteLength];
            original.CopyDataTo(before); converted.ToTernary2().CopyDataTo(after);
            Assert.Equal(before, after);
            float[] expected = new float[sample.Rows], actual = new float[sample.Rows];
            Ternary2MatVec.ExecuteScalar(original, sample.Input, expected);
            TernaryBitplaneMatVec.ExecuteScalar(converted, sample.Input, actual);
            Assert.Equal(expected, actual);
            for (int i = 0; i < 20; i++) TernaryBitplaneMatVec.ExecuteScalar(converted, sample.Input, actual);
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 20; i++) TernaryBitplaneMatVec.ExecuteScalar(converted, sample.Input, actual);
            Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - allocated);
        }
    }

    [Fact]
    public void FileHeaderIsIndependentAndTruncationsReject()
    {
        byte[] payload = new byte[20];
        BinaryPrimitives.WriteSingleLittleEndian(payload, 0.5f); payload[4] = 0x89; payload[12] = 0x14;
        var matrix = TernaryBitplaneMatrix.FromPacked(1, 8, 64, payload);
        using var directory = new MappedFileFixture(1, 0, [0]);
        string path = directory.Path + ".stb1";
        matrix.Save(path);
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal("STB1"u8.ToArray(), bytes[..4]);
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(4)));
        Assert.Equal(20, BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(24)));
        var loaded = TernaryBitplaneMatrix.Load(path);
        byte[] copy = new byte[20]; loaded.CopyDataTo(copy); Assert.Equal(payload, copy);
        for (int length = 0; length < bytes.Length; length++)
        {
            File.WriteAllBytes(path, bytes[..length]);
            Assert.Throws<InvalidDataException>(() => TernaryBitplaneMatrix.Load(path));
        }
    }
}

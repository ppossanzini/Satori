using System.Buffers.Binary;
using Satori.Core;
using Satori.Validation;

namespace Satori.Numerics.Tests;

public class MappedTernary2Tests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void MappedKernelsMatchOwnedKernelsAcrossCriticalCorpus(int blockSize)
    {
        foreach (var sample in NumericalCorpus.CriticalCases(blockSize))
        {
            var matrix = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns, blockSize);
            using var fixture = new MappedFileFixture(1, 0, [0]);
            File.Delete(fixture.Path);
            matrix.Save(fixture.Path);
            using var mapped = new MappedTernary2Matrix(fixture.Path);
            float[] expected = new float[sample.Rows], actual = new float[sample.Rows];
            Ternary2MatVec.ExecuteScalar(matrix, sample.Input, expected);
            mapped.ExecuteScalar(sample.Input, actual);
            Assert.Equal(expected, actual);
            Ternary2MatVec.Execute(matrix, sample.Input, expected);
            mapped.Execute(sample.Input, actual);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void RejectsInvalidHeaderLengthAndPayloadBeforeExecuting()
    {
        var matrix = Ternary2Matrix.Quantize([1f], 1, 1);
        using var valid = new MappedFileFixture(1, 0, [0]);
        File.Delete(valid.Path);
        matrix.Save(valid.Path);
        byte[] original = File.ReadAllBytes(valid.Path);
        Assert.Equal(new byte[]
        {
            0x53, 0x54, 0x51, 0x32, 1, 0, 32, 0,
            1, 0, 0, 0, 1, 0, 0, 0,
            64, 0, 0, 0, 0, 0, 0, 0,
            20, 0, 0, 0, 0, 0, 0, 0
        }, original[..32]);
        byte[] trailing = [.. original, 0];
        var corruptions = new List<byte[]> { original[..10], original[..^1], trailing };
        foreach (int index in new[] { 0, 4, 6, 8, 16, 20, 24 })
        {
            byte[] copy = (byte[])original.Clone();
            copy[index] ^= 0xFF;
            corruptions.Add(copy);
        }
        byte[] invalidCode = (byte[])original.Clone();
        invalidCode[36] = 3;
        corruptions.Add(invalidCode);
        byte[] invalidPadding = (byte[])original.Clone();
        invalidPadding[36] |= 4;
        corruptions.Add(invalidPadding);
        byte[] invalidScale = (byte[])original.Clone();
        BinaryPrimitives.WriteSingleLittleEndian(invalidScale.AsSpan(32), float.NaN);
        corruptions.Add(invalidScale);
        foreach (byte[] data in corruptions)
        {
            using var fixture = new MappedFileFixture(data.Length, 0, data);
            Assert.Throws<InvalidDataException>(() => new MappedTernary2Matrix(fixture.Path));
        }
    }

    [Fact]
    public void MappedExecutionAllocatesNothingAndRejectsAccessAfterDispose()
    {
        var sample = NumericalCorpus.RandomMatrix(7, 257);
        var matrix = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns);
        using var fixture = new MappedFileFixture(1, 0, [0]);
        File.Delete(fixture.Path);
        matrix.Save(fixture.Path);
        using var mapped = new MappedTernary2Matrix(fixture.Path);
        float[] output = new float[sample.Rows];
        for (int i = 0; i < 100; i++) mapped.Execute(sample.Input, output);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) mapped.Execute(sample.Input, output);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
        mapped.Dispose();
        Assert.Throws<ObjectDisposedException>(() => mapped.Execute(sample.Input, output));
    }
}

using System.Buffers.Binary;
using Satori.Formats.Gguf;
using Satori.Core;

namespace Satori.Numerics.Tests;

public sealed class TransformerWeightsTests
{
    [Fact]
    public void GatherReadsSelectedRowAndBorrowedLifetimeIsEnforced()
    {
        byte[] prefix = GgufContainerFixture.Prefix(32, extras: 1, metadata: writer =>
        {
            GgufContainerFixture.String(writer,"general.quantization_version"); writer.Write(4u); writer.Write(2u);
        }, tensors: [("embedding", [32,2], 8, 0), ("norm", [2], 0, 96)]);
        byte[] bytes = new byte[prefix.Length + 128]; prefix.CopyTo(bytes,0);
        for (int row = 0; row < 2; row++)
        {
            bytes[prefix.Length + row*34 + 1] = 0x3C; // binary16 1
            for (int j = 0; j < 32; j++) bytes[prefix.Length + row*34+2+j] = (byte)(row*32+j);
        }
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(prefix.Length+96), 1.25f);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(prefix.Length+100), -2.5f);
        using var fixture = new MappedFileFixture(bytes.Length,0,bytes);
        var file = new GgufFile(fixture.Path);
        var matrix = new GgufQuantizedMatrix(file,"embedding");
        var norm = new GgufFloatVector(file,"norm");
        float[] output = new float[32];
        matrix.ReadRow(1,output);
        for (int j = 0; j < 32; j++) Assert.Equal(32+j,output[j]);
        Assert.Throws<ArgumentOutOfRangeException>(() => matrix.ReadRow(2,output));
        matrix.ReadRow(0,output); // warm callback
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) matrix.ReadRow(1,output);
        Assert.Equal(0,GC.GetAllocatedBytesForCurrentThread()-before);
        file.Dispose();
        Assert.Equal(new float[] {1.25f,-2.5f},norm.Values.ToArray());
        Assert.Throws<ObjectDisposedException>(() => matrix.ReadRow(0,output));
    }

    [Fact]
    public void NonFiniteNormalizationWeightIsRejected()
    {
        byte[] prefix = GgufContainerFixture.Prefix(32,tensors: [("norm",[1],0,0)]);
        byte[] bytes = new byte[prefix.Length+32]; prefix.CopyTo(bytes,0);
        BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(prefix.Length),float.NaN);
        using var fixture = new MappedFileFixture(bytes.Length,0,bytes);
        using var file = new GgufFile(fixture.Path);
        Assert.Throws<InvalidDataException>(() => new GgufFloatVector(file,"norm"));
    }

    [Fact]
    public void OperatorsAllocateNothingAndSingleTokenAttentionReturnsValue()
    {
        float[] q = new float[288], k = new float[288], v = Enumerable.Range(0,288).Select(i => (float)i).ToArray();
        float[] weights = Enumerable.Repeat(1f,288).ToArray(), output = new float[288];
        double[] scores = new double[1];
        void Run()
        {
            TransformerOps.RmsNorm(q,weights,1e-5f,output);
            TransformerOps.Rope(output,48,127);
            TransformerOps.SwiGlu(q,v,output);
            TransformerOps.Add(q,v,output);
            TransformerOps.Attention(q,k,v,48,1,scores,output);
        }
        Run();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) Run();
        long allocated = GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.Equal(0,allocated);
        Assert.Equal(v,output);
    }
}

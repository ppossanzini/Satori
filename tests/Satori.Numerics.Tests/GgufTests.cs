using System.Buffers.Binary;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class GgufTests
{
    [Fact]
    public void RepeatedMappedStringReadsKeepNamesAndScalarTypesStable()
    {
        string path = Path.Combine(Root,"fixtures/gguf/v1/q8_0.gguf");
        for (int i = 0; i < 1000; i++)
        {
            using var file = new GgufFile(path);
            Assert.Equal("llama",file.ReadScalar("general.architecture"));
            Assert.Equal(2u,file.ReadScalar("general.quantization_version"));
            Assert.Equal(8u,file.Tensors["weights"].Type);
        }
    }

    internal static string Root
    {
        get
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Satori.sln"))) dir = dir.Parent;
            return dir!.FullName;
        }
    }

    [Theory]
    [InlineData("q4_0", 2, 18, 160)]
    [InlineData("q8_0", 8, 34, 160)]
    [InlineData("f32", 0, 12, 128)]
    [InlineData("f16", 1, 6, 128)]
    public void GoldenCatalogueAndScopedBounds(string name, uint type, long length, long offset)
    {
        using var file = new GgufFile(Path.Combine(Root, "fixtures/gguf/v1", name + ".gguf"));
        var tensor = Assert.Single(file.Tensors).Value;
        Assert.Equal(type, tensor.Type);
        Assert.Equal(length, tensor.ByteLength);
        Assert.Equal(offset, tensor.Offset);
        Assert.Equal("llama", file.ReadScalar("general.architecture"));
        Assert.Throws<ArgumentOutOfRangeException>(() => file.ReadTensor("weights", length, 1, static (_, _) => { }, 0));
        file.ReadTensor("weights", length, 0, static (bytes, _) => Assert.Empty(bytes.ToArray()), 0);
        file.Dispose();
        Assert.Throws<ObjectDisposedException>(() => file.ReadTensor("weights", 0, 1, static (_, _) => { }, 0));
    }

    [Fact]
    public void EveryTruncationRejectsWithoutPublishingCatalogue()
    {
        byte[] golden = File.ReadAllBytes(Path.Combine(Root, "fixtures/gguf/v1/q8_0.gguf"));
        for (int n = 0; n < golden.Length; n++)
        {
            using var temp = new MappedFileFixture(n, 0, golden[..n]);
            Assert.Throws<InvalidDataException>(() => new GgufFile(temp.Path));
        }
    }

    [Fact]
    public void MutatedStructureAndUnsupportedVersionReject()
    {
        byte[] golden = File.ReadAllBytes(Path.Combine(Root, "fixtures/gguf/v1/q8_0.gguf"));
        foreach (int index in new[] { 0, 32, 45, 137 })
        {
            byte[] bytes = (byte[])golden.Clone();
            bytes[index] = 0xff;
            using var temp = new MappedFileFixture(bytes.Length, 0, bytes);
            Assert.Throws<InvalidDataException>(() => new GgufFile(temp.Path));
        }
        BinaryPrimitives.WriteUInt32LittleEndian(golden.AsSpan(4), 2);
        using var unsupported = new MappedFileFixture(golden.Length, 0, golden);
        Assert.Throws<NotSupportedException>(() => new GgufFile(unsupported.Path));
    }

    [Fact]
    public void MappingCallbackSurvivesDisposalAndException()
    {
        using var file = new GgufFile(Path.Combine(Root, "fixtures/gguf/v1/q8_0.gguf"));
        file.ReadTensor("weights", 0, 34, (bytes, _) =>
        {
            file.Dispose();
            Assert.Equal(0x38, bytes[1]);
        }, 0);
        Assert.Throws<ObjectDisposedException>(() => file.ReadMetadata("general.architecture", static (_, _) => { }, 0));
    }
}

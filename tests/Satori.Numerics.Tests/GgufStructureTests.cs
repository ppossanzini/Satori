using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class GgufStructureTests
{
    [Theory]
    [InlineData(8u)]
    [InlineData(64u)]
    [InlineData(4096u)]
    public void MixedTensorsAllScalarTypesAndFlatArrays(uint alignment)
    {
        byte[] prefix = GgufContainerFixture.Prefix(alignment, writer =>
        {
            for (uint type = 0; type <= 12; type++)
            {
                GgufContainerFixture.String(writer, $"custom.value_{type}"); writer.Write(type);
                WriteValue(writer, type);
            }
            GgufContainerFixture.String(writer, "custom.empty"); writer.Write(9u); writer.Write(8u); writer.Write(0ul);
            GgufContainerFixture.String(writer, "custom.strings"); writer.Write(9u); writer.Write(8u); writer.Write(2ul);
            GgufContainerFixture.String(writer, "è\0漢"); GgufContainerFixture.String(writer, "");
        }, 15);
        using var temp = new MappedFileFixture(prefix.Length + alignment * 2 + 3, 0, prefix);
        using var file = new GgufFile(temp.Path);
        Assert.Equal(2, file.Tensors.Count);
        Assert.Equal(17, file.Metadata.Count);
        Assert.Equal("è\0漢", file.ReadScalar("custom.value_8"));
        Assert.Equal((uint)17, file.ReadScalar("custom.value_4"));
        Assert.Equal(-17L, file.ReadScalar("custom.value_11"));
        Assert.True(double.IsNaN((double)file.ReadScalar("custom.value_12")));
        Assert.Equal(0, file.Metadata["custom.empty"].ElementCount);
        Assert.Equal(2, file.Metadata["custom.strings"].ElementCount);
        Assert.Equal(prefix.Length + alignment, file.Tensors["second"].Offset);
    }

    private static void WriteValue(BinaryWriter writer, uint type)
    {
        switch (type)
        {
            case 0: writer.Write((byte)17); break;
            case 1: writer.Write((sbyte)-17); break;
            case 2: writer.Write((ushort)17); break;
            case 3: writer.Write((short)-17); break;
            case 4: writer.Write(17u); break;
            case 5: writer.Write(-17); break;
            case 6: writer.Write(float.PositiveInfinity); break;
            case 7: writer.Write((byte)1); break;
            case 8: GgufContainerFixture.String(writer, "è\0漢"); break;
            case 9: writer.Write(7u); writer.Write(2ul); writer.Write((byte)0); writer.Write((byte)1); break;
            case 10: writer.Write(17ul); break;
            case 11: writer.Write(-17L); break;
            case 12: writer.Write(double.NaN); break;
        }
    }

    [Fact]
    public void OversizedSparseFileRejectsBeforeMapping()
    {
        using var temp = new MappedFileFixture((1L << 40) + 1, 0, []);
        var error = Assert.Throws<NotSupportedException>(() => new GgufFile(temp.Path));
        Assert.Contains("File limit", error.Message);
    }

    [Fact]
    public void SparseTensorBeyondTwoGiBUsesInt64Windows()
    {
        const ulong firstLength = 1ul << 31;
        byte[] prefix = GgufContainerFixture.Prefix(32, tensors:
            [("large", [firstLength / 4], 0, 0), ("tail", [1], 0, firstLength)]);
        using var temp = new MappedFileFixture(prefix.Length + (long)firstLength + 32, 0, prefix);
        using (var stream = new FileStream(temp.Path, FileMode.Open, FileAccess.Write))
        {
            stream.Position = prefix.Length + (long)firstLength;
            stream.Write([1, 2, 3, 4]);
        }
        using var file = new GgufFile(temp.Path);
        file.ReadTensor("tail", 0, 4, static (bytes, _) => Assert.Equal(new byte[] { 1, 2, 3, 4 }, bytes.ToArray()), 0);
        Assert.Throws<ArgumentOutOfRangeException>(() => file.ReadTensor("large", (long)firstLength - 1, 2, static (_, _) => { }, 0));
    }

    [Theory]
    [InlineData("bool")]
    [InlineData("key")]
    [InlineData("duplicate")]
    [InlineData("utf8")]
    [InlineData("typed")]
    public void InvalidMetadataRejects(string defect)
    {
        byte[] prefix = GgufContainerFixture.Prefix(32, writer =>
        {
            GgufContainerFixture.String(writer, defect switch
            {
                "key" => "BAD.key", "duplicate" => "general.architecture", "typed" => "general.file_type", _ => "custom.test"
            });
            if (defect == "utf8") { writer.Write(8u); writer.Write(1ul); writer.Write((byte)0xff); }
            else { writer.Write(7u); writer.Write(defect == "bool" ? (byte)2 : (byte)1); }
        }, 1);
        using var temp = new MappedFileFixture(prefix.Length + 64, 0, prefix);
        Assert.Throws<InvalidDataException>(() => new GgufFile(temp.Path));
    }

    [Theory]
    [InlineData("overlap")]
    [InlineData("overflow")]
    [InlineData("zero")]
    [InlineData("unaligned")]
    [InlineData("duplicate")]
    public void InvalidTensorStructureRejects(string defect)
    {
        var tensors = defect switch
        {
            "overlap" => new[] { ("a", new ulong[] { 32 }, 0u, 0ul), ("b", new ulong[] { 1 }, 0u, 32ul) },
            "overflow" => [("a", new ulong[] { long.MaxValue, 2 }, 0u, 0ul)],
            "zero" => [("a", new ulong[] { 0 }, 0u, 0ul)],
            "unaligned" => [("a", new ulong[] { 1 }, 0u, 1ul)],
            _ => [("a", new ulong[] { 1 }, 0u, 0ul), ("a", new ulong[] { 1 }, 0u, 32ul)]
        };
        byte[] prefix = GgufContainerFixture.Prefix(32, tensors: tensors);
        using var temp = new MappedFileFixture(prefix.Length + 256, 0, prefix);
        Assert.Throws<InvalidDataException>(() => new GgufFile(temp.Path));
    }

    [Fact]
    public void GapAndReversePhysicalOrderAreUnsupportedRatherThanCorrupt()
    {
        foreach (var offsets in new[] { new ulong[] { 32, 64 }, new ulong[] { 32, 0 } })
        {
            byte[] prefix = GgufContainerFixture.Prefix(32, tensors:
                [("a", [1], 0, offsets[0]), ("b", [1], 0, offsets[1])]);
            using var temp = new MappedFileFixture(prefix.Length + 128, 0, prefix);
            Assert.Throws<NotSupportedException>(() => new GgufFile(temp.Path));
        }
    }
}

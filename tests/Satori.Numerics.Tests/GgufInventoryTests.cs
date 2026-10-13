using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS01: the descriptor inventory must never diverge from the validated catalogue on supported files,
/// and must report unsupported tensor types explicitly instead of rejecting the file.
/// </summary>
public sealed class GgufInventoryTests
{
    [Theory]
    [InlineData("q4_0")]
    [InlineData("q8_0")]
    [InlineData("f32")]
    [InlineData("f16")]
    public void DescriptorScanAgreesWithValidatedCatalogue(string name)
    {
        string path = Path.Combine(GgufTests.Root, "fixtures/gguf/v1", name + ".gguf");
        var inventory = GgufInventory.Scan(path);
        using var file = new GgufFile(path);

        Assert.True(inventory.ProfileCompatible);
        Assert.Equal(file.Tensors.Count, inventory.Tensors.Count);
        Assert.Equal(file.Metadata.Count, inventory.Metadata.Count);
        Assert.Equal(file.DataOffset, inventory.DataOffset);
        Assert.Equal("llama", inventory.Architecture);
        foreach (var descriptor in inventory.Tensors)
        {
            var tensor = file.Tensors[descriptor.Name];
            Assert.Equal(tensor.Type, descriptor.TypeId);
            Assert.Equal(tensor.Dimensions, descriptor.Dimensions);
            Assert.Equal(tensor.ByteLength, descriptor.ByteLength);
            Assert.Equal(tensor.Offset, inventory.DataOffset + descriptor.Offset);
        }
    }

    [Fact]
    public void UnsupportedTensorTypesAreReportedNotRejected()
    {
        byte[] bytes = GgufContainerFixture.Prefix(32, tensors:
        [
            ("q4k", [256, 2], 12, 0),
            ("q6k", [256, 1], 14, 288),
        ]);
        using var temp = new MappedFileFixture(bytes.Length, 0, bytes);
        var inventory = GgufInventory.Scan(temp.Path);

        Assert.False(inventory.ProfileCompatible);
        Assert.Equal(["Q4_K(12)", "Q6_K(14)"], inventory.UnsupportedTypes.OrderBy(x => x));
        Assert.All(inventory.Tensors, t => Assert.False(t.Supported));
        Assert.True(inventory.TotalTensorBytesExact);
        Assert.Equal(288 + 210, inventory.TotalTensorBytes);
    }
}

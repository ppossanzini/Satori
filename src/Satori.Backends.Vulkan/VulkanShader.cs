using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;

namespace Satori.Backends.Vulkan;

internal static class VulkanShader
{
    internal static byte[] Load(string name = "transform")
    {
        var assembly = typeof(VulkanShader).Assembly;
        using var binary = assembly.GetManifestResourceStream($"Satori.Backends.Vulkan.Shaders.{name}.spv")
            ?? throw new InvalidDataException("Missing embedded compute shader.");
        using var memory = new MemoryStream();
        binary.CopyTo(memory);
        byte[] bytes = memory.ToArray();
        using var manifest = assembly.GetManifestResourceStream($"Satori.Backends.Vulkan.Shaders.{name}.json")
            ?? throw new InvalidDataException("Missing embedded shader manifest.");
        using var json = JsonDocument.Parse(manifest);
        if (bytes.Length < 20 || bytes.Length%4 != 0 || BinaryPrimitives.ReadUInt32LittleEndian(bytes)!=0x07230203
            || BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(4))!=0x00010300
            || Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant()!=json.RootElement.GetProperty("binary_sha256").GetString())
            throw new InvalidDataException("Embedded shader does not match its SPIR-V manifest.");
        return bytes;
    }
}

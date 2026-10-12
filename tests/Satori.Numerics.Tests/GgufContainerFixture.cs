using System.Text;

namespace Satori.Numerics.Tests;

internal static class GgufContainerFixture
{
    internal static void String(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    internal static byte[] Prefix(uint alignment, Action<BinaryWriter>? metadata = null, int extras = 0,
        (string Name, ulong[] Shape, uint Type, ulong Offset)[]? tensors = null)
    {
        tensors ??= [("first", [2], 0, 0), ("second", [2], 1, alignment)];
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        writer.Write("GGUF"u8);
        writer.Write(3u);
        writer.Write((ulong)tensors.Length);
        writer.Write((ulong)(2 + extras));
        String(writer, "general.architecture"); writer.Write(8u); String(writer, "llama");
        String(writer, "general.alignment"); writer.Write(4u); writer.Write(alignment);
        metadata?.Invoke(writer);
        foreach (var tensor in tensors)
        {
            String(writer, tensor.Name); writer.Write((uint)tensor.Shape.Length);
            foreach (ulong d in tensor.Shape) writer.Write(d);
            writer.Write(tensor.Type); writer.Write(tensor.Offset);
        }
        while (stream.Length % alignment != 0) writer.Write((byte)0);
        return stream.ToArray();
    }
}

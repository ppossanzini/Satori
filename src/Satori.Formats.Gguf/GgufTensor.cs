using System.Collections.ObjectModel;

namespace Satori.Formats.Gguf;

public sealed record GgufTensor
{
    internal GgufTensor(string name, long[] dimensions, uint type, long offset, long length)
    {
        Name = name;
        Dimensions = Array.AsReadOnly(dimensions);
        Type = type;
        Offset = offset;
        ByteLength = length;
    }

    public string Name { get; }
    public ReadOnlyCollection<long> Dimensions { get; }
    public uint Type { get; }
    public long Offset { get; }
    public long ByteLength { get; }
}

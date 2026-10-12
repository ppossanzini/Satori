namespace Satori.Formats.Gguf;

public sealed record GgufMetadata(string Key, uint Type, long Offset, long ByteLength,
    uint? ElementType, long? ElementCount);

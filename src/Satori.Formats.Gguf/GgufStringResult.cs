namespace Satori.Formats.Gguf;

/// <summary>Receives a decoded string from a scoped mapped read.</summary>
internal sealed class GgufStringResult
{
    internal string Value { get; set; } = "";
}

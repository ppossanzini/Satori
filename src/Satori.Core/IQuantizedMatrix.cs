namespace Satori.Core;

/// <summary>Owned executable weights with synchronous vector input and output.</summary>
public interface IQuantizedMatrix : IDisposable
{
    /// <summary>Commits output only on success; a failed execution leaves output unchanged.</summary>
    void Execute(ReadOnlySpan<float> input, Span<float> output);
}

namespace Satori.Core;

/// <summary>Creates independent executable weights from a callback-scoped serialized payload.</summary>
public interface IQuantizedMatrixFactory
{
    IQuantizedMatrix Create(ReadOnlySpan<byte> payload, QuantizedMatrixShape shape);
}

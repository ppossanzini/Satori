namespace Satori.Core.Model;

/// <summary>
/// Read access to the weights a dense forward pass needs. Core depends only on this interface; the GGUF
/// module provides the implementation. All methods are synchronous and allocation-free on the hot path.
/// </summary>
/// <remarks>
/// The implementation owns the storage lifetime. A store obtained for a definition must stay valid for the
/// whole forward pass; disposing the underlying container while a pass is in flight is a caller error.
/// </remarks>
public interface IModelWeights : IDisposable
{
    /// <summary>True when the named tensor is present in the container.</summary>
    bool Has(string name);

    /// <summary>Computes <c>output = weights * input</c> for the named rank-2 quantized tensor.</summary>
    /// <remarks><paramref name="input"/> length must equal the tensor column count and
    /// <paramref name="output"/> length the row count.</remarks>
    void Project(string name, ReadOnlySpan<float> input, Span<float> output);

    /// <summary>Returns the named rank-1 F32 vector (a normalization weight or a projection bias).</summary>
    ReadOnlySpan<float> Vector(string name);

    /// <summary>Writes the embedding row of the named matrix for <paramref name="token"/>.</summary>
    void EmbedRow(string name, int token, Span<float> output);
}

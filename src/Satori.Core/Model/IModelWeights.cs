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

    /// <summary>
    /// Computes the named rank-2 tensor against <paramref name="positions"/> input vectors at once.
    /// <paramref name="input"/> holds that many vectors of the tensor's column count, laid out position after
    /// position, and <paramref name="output"/> receives as many vectors of its row count in the same layout.
    /// </summary>
    /// <remarks>
    /// This exists so a prompt does not read the packed weights once per token. An implementation may split
    /// the rows over <paramref name="threads"/> workers; the result must not depend on that split.
    /// <para>
    /// <paramref name="cancellationToken"/> is observed between the pieces of work that split produces, so a
    /// request to stop waits at most one piece and not one whole projection. A projection of the largest shape
    /// in the model is the longest uninterruptible step of a layer, and bounding the delay by a piece rather
    /// than by the projection is what makes the delay a property of the design instead of of luck.
    /// </para>
    /// </remarks>
    void ProjectBatch(string name, ReadOnlySpan<float> input, int positions, Span<float> output, int threads,
        CancellationToken cancellationToken = default);

    /// <summary>Returns the named rank-1 F32 vector (a normalization weight or a projection bias).</summary>
    ReadOnlySpan<float> Vector(string name);

    /// <summary>Writes the embedding row of the named matrix for <paramref name="token"/>.</summary>
    void EmbedRow(string name, int token, Span<float> output);
}

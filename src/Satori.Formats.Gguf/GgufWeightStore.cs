using Satori.Core.Model;

namespace Satori.Formats.Gguf;

/// <summary>
/// <see cref="IModelWeights"/> over a validated GGUF container. Projections run the shared quantized MatVec
/// directly on mapped bytes; vectors are decoded once and cached; embedding rows are decoded on demand.
/// </summary>
/// <remarks>
/// The store borrows the <see cref="GgufFile"/> lifetime and never disposes it: the caller owns the
/// container. Matrix handles are created lazily but allocate no payload; they validate the payload once.
/// </remarks>
public sealed class GgufWeightStore : IModelWeights
{
    private readonly GgufFile _file;
    private readonly Dictionary<string, GgufQuantizedMatrix> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgufFloatVector> _vectors = new(StringComparer.Ordinal);

    public GgufWeightStore(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        _file = file;
    }

    public bool Has(string name) => _file.Tensors.ContainsKey(name);

    public void Project(string name, ReadOnlySpan<float> input, Span<float> output) =>
        Matrix(name).Execute(input, output);

    public ReadOnlySpan<float> Vector(string name) => VectorOf(name).Values;

    public void EmbedRow(string name, int token, Span<float> output) =>
        Matrix(name).ReadRow(token, output);

    /// <summary>The store does not own the container; disposal is a no-op for symmetry with the contract.</summary>
    public void Dispose() { }

    private GgufQuantizedMatrix Matrix(string name)
    {
        if (_matrices.TryGetValue(name, out var matrix)) return matrix;
        matrix = new GgufQuantizedMatrix(_file, name);
        _matrices[name] = matrix;
        return matrix;
    }

    private GgufFloatVector VectorOf(string name)
    {
        if (_vectors.TryGetValue(name, out var vector)) return vector;
        vector = new GgufFloatVector(_file, name);
        _vectors[name] = vector;
        return vector;
    }
}

using System.Collections.Concurrent;
using Satori.Core.Model;

namespace Satori.Formats.Gguf;

/// <summary>
/// <see cref="IModelWeights"/> over a validated GGUF container. Projections run the shared quantized MatVec
/// directly on mapped bytes; vectors are decoded once and cached; embedding rows are decoded on demand.
/// </summary>
/// <remarks>
/// The store borrows the <see cref="GgufFile"/> lifetime and never disposes it: the caller owns the
/// container. Matrix handles are created lazily but allocate no payload; they validate the payload once.
/// <para>
/// A batched prefill projects from several threads at once, so handle resolution is thread-safe and each
/// handle is built exactly once: the caches hold <see cref="Lazy{T}"/> entries, which lets a losing racer
/// reuse the winner's handle instead of decoding the same tensor a second time.
/// </para>
/// </remarks>
public sealed class GgufWeightStore : IModelWeights
{
    private readonly GgufFile _file;
    private readonly ConcurrentDictionary<string, Lazy<GgufQuantizedMatrix>> _matrices = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, Lazy<GgufFloatVector>> _vectors = new(StringComparer.Ordinal);

    public GgufWeightStore(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        _file = file;
    }

    public bool Has(string name) => _file.Tensors.ContainsKey(name);

    public void Project(string name, ReadOnlySpan<float> input, Span<float> output) =>
        Matrix(name).Execute(input, output);

    public void ProjectBatch(string name, ReadOnlySpan<float> input, int positions, Span<float> output, int threads) =>
        Matrix(name).ExecuteBatch(input, positions, output, threads);

    public ReadOnlySpan<float> Vector(string name) => VectorOf(name).Values;

    public void EmbedRow(string name, int token, Span<float> output) =>
        Matrix(name).ReadRow(token, output);

    /// <summary>The store does not own the container; disposal is a no-op for symmetry with the contract.</summary>
    public void Dispose() { }

    private GgufQuantizedMatrix Matrix(string name) => _matrices.GetOrAdd(name,
        static (key, file) => new Lazy<GgufQuantizedMatrix>(() => new GgufQuantizedMatrix(file, key),
            LazyThreadSafetyMode.ExecutionAndPublication), _file).Value;

    private GgufFloatVector VectorOf(string name) => _vectors.GetOrAdd(name,
        static (key, file) => new Lazy<GgufFloatVector>(() => new GgufFloatVector(file, key),
            LazyThreadSafetyMode.ExecutionAndPublication), _file).Value;
}

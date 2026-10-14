using Satori.Core.Model;

namespace Satori.Numerics.Tests;

/// <summary>
/// Deterministic in-memory <see cref="IModelWeights"/> for offline forward/session tests. Matrices are dense
/// row-major F32, projections are a plain FP32 dot, so the incremental-versus-recomputation invariant does
/// not depend on any quantized kernel.
/// </summary>
internal sealed class SyntheticWeights : IModelWeights
{
    private readonly Dictionary<string, float[]> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (int Rows, int Columns)> _shapes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _vectors = new(StringComparer.Ordinal);

    public SyntheticWeights(ModelShape shape, bool bias, bool headNorm, int seed = 12345)
    {
        int embedding = shape.EmbeddingLength;
        int feedForward = shape.FeedForwardLength;
        int qWidth = shape.QueryWidth;
        int kvWidth = shape.KvWidth;
        var random = new Random(seed);

        AddMatrix("token_embd.weight", shape.VocabularySize, embedding, random);
        if (!shape.TiedEmbeddings) AddMatrix("output.weight", shape.VocabularySize, embedding, random);
        AddVector("output_norm.weight", embedding, random, unit: true);

        for (int layer = 0; layer < shape.Layers; layer++)
        {
            string prefix = $"blk.{layer}.";
            AddVector(prefix + "attn_norm.weight", embedding, random, unit: true);
            AddMatrix(prefix + "attn_q.weight", qWidth, embedding, random);
            AddMatrix(prefix + "attn_k.weight", kvWidth, embedding, random);
            AddMatrix(prefix + "attn_v.weight", kvWidth, embedding, random);
            AddMatrix(prefix + "attn_output.weight", embedding, qWidth, random);
            AddVector(prefix + "ffn_norm.weight", embedding, random, unit: true);
            AddMatrix(prefix + "ffn_gate.weight", feedForward, embedding, random);
            AddMatrix(prefix + "ffn_up.weight", feedForward, embedding, random);
            AddMatrix(prefix + "ffn_down.weight", embedding, feedForward, random);
            if (bias)
            {
                AddVector(prefix + "attn_q.bias", qWidth, random);
                AddVector(prefix + "attn_k.bias", kvWidth, random);
                AddVector(prefix + "attn_v.bias", kvWidth, random);
            }
            if (headNorm)
            {
                AddVector(prefix + "attn_q_norm.weight", shape.HeadDim, random, unit: true);
                AddVector(prefix + "attn_k_norm.weight", shape.HeadDim, random, unit: true);
            }
        }
    }

    public bool Has(string name) => _matrices.ContainsKey(name) || _vectors.ContainsKey(name);

    public void Project(string name, ReadOnlySpan<float> input, Span<float> output)
    {
        var (rows, columns) = _shapes[name];
        if (input.Length != columns || output.Length != rows) throw new ArgumentException($"Shape mismatch for {name}.");
        float[] matrix = _matrices[name];
        for (int row = 0; row < rows; row++)
        {
            double sum = 0;
            int offset = row * columns;
            for (int column = 0; column < columns; column++) sum += (double)matrix[offset + column] * input[column];
            output[row] = (float)sum;
        }
    }

    /// <summary>
    /// Accumulates each element exactly as <see cref="Project"/> does, in the same order, so a batched pass
    /// and a single-position pass are bit-identical here and the batched-prefill test stays exact. Rows are
    /// split over <paramref name="threads"/> workers exactly as the quantized kernel splits them, and each
    /// worker writes its own rows, so the split cannot change a value.
    /// </summary>
    public void ProjectBatch(string name, ReadOnlySpan<float> input, int positions, Span<float> output, int threads)
    {
        var (rows, columns) = _shapes[name];
        if (input.Length != positions * columns) throw new ArgumentException($"Input must hold {positions} vectors for {name}.");
        if (output.Length != positions * rows) throw new ArgumentException($"Output must hold {positions} vectors for {name}.");
        float[] matrix = _matrices[name];
        int workers = Math.Min(threads, rows);
        int stride = (rows + workers - 1) / workers;
        for (int worker = 0; worker < workers; worker++)
        {
            int begin = worker * stride;
            int end = Math.Min(begin + stride, rows);
            for (int row = begin; row < end; row++)
            {
                int offset = row * columns;
                for (int position = 0; position < positions; position++)
                {
                    double sum = 0;
                    int source = position * columns;
                    for (int column = 0; column < columns; column++) sum += (double)matrix[offset + column] * input[source + column];
                    output[position * rows + row] = (float)sum;
                }
            }
        }
    }

    public ReadOnlySpan<float> Vector(string name) => _vectors[name];

    public void EmbedRow(string name, int token, Span<float> output)
    {
        var (_, columns) = _shapes[name];
        _matrices[name].AsSpan(token * columns, columns).CopyTo(output);
    }

    public void Dispose() { }

    private void AddMatrix(string name, int rows, int columns, Random random)
    {
        var values = new float[checked(rows * columns)];
        for (int i = 0; i < values.Length; i++) values[i] = (float)((random.NextDouble() - 0.5) * 0.5);
        _matrices[name] = values;
        _shapes[name] = (rows, columns);
    }

    private void AddVector(string name, int length, Random random, bool unit = false)
    {
        var values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = unit ? (float)(1.0 + (random.NextDouble() - 0.5) * 0.1) : (float)((random.NextDouble() - 0.5) * 0.5);
        _vectors[name] = values;
    }
}

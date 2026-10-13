namespace Satori.Core.Model;

/// <summary>Observes an intermediate tensor after a named stage. <paramref name="layer"/> is <c>-1</c> for a
/// final stage and <paramref name="position"/> is <c>-1</c> when the value is not position-specific. The
/// span is only valid for the duration of the callback.</summary>
public delegate void ForwardObserver(int layer, int position, string stage, ReadOnlySpan<float> values);

/// <summary>
/// Dense decoder forward pass for the qwen2/qwen3 families. It covers embedding, per-layer grouped-query
/// attention with half-split RoPE and optional Q/K per-head normalization, a SwiGLU feed-forward block, the
/// final normalization and the output projection.
/// </summary>
/// <remarks>
/// Every dimension comes from the validated <see cref="ModelShape"/>; nothing is a nominal constant. Every
/// operator is the shared <see cref="TransformerOps"/> implementation and every projection is the shared
/// quantized MatVec, so a new tensor format is supported by the kernel registry, not by this class. The
/// whole token sequence is processed at once; incremental state is a later task.
/// </remarks>
public sealed class DenseForward
{
    /// <summary>Embedding-matrix name; also the output projection when the embeddings are tied.</summary>
    public const string TokenEmbedding = "token_embd.weight";

    private readonly ModelShape _shape;
    private readonly IModelWeights _weights;
    private readonly LayerNames[] _layers;
    private readonly string _output;
    private readonly string _outputNorm;
    private readonly int _embedding;
    private readonly int _feedForward;
    private readonly int _qWidth;
    private readonly int _kvWidth;
    private readonly int _headDim;
    private readonly float _epsilon;
    private readonly double _ropeBase;

    public DenseForward(ModelShape shape, IModelWeights weights)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(weights);
        _shape = shape;
        _weights = weights;
        _embedding = shape.EmbeddingLength;
        _feedForward = shape.FeedForwardLength;
        _qWidth = checked(shape.HeadCount * shape.HeadDim);
        _kvWidth = checked(shape.HeadCountKv * shape.HeadDim);
        _headDim = shape.HeadDim;
        _epsilon = (float)shape.RmsNormEpsilon;
        _ropeBase = shape.RopeBase;
        _outputNorm = "output_norm.weight";
        _output = shape.TiedEmbeddings ? TokenEmbedding : "output.weight";
        _layers = new LayerNames[shape.Layers];
        for (int layer = 0; layer < shape.Layers; layer++) _layers[layer] = new LayerNames(layer, weights);
    }

    public ModelShape Shape => _shape;

    /// <summary>Logits of the final position for the given token sequence.</summary>
    public float[] Logits(ReadOnlySpan<int> tokens) => Logits(tokens, null);

    /// <summary>Logits of the final position, optionally reporting intermediate tensors.</summary>
    public float[] Logits(ReadOnlySpan<int> tokens, ForwardObserver? observe)
    {
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        int count = tokens.Length;

        float[] hidden = new float[checked(count * _embedding)];
        for (int position = 0; position < count; position++)
        {
            int token = tokens[position];
            if ((uint)token >= (uint)_shape.VocabularySize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token {token} is outside the vocabulary.");
            _weights.EmbedRow(TokenEmbedding, token, hidden.AsSpan(position * _embedding, _embedding));
        }

        float[] normed = new float[_embedding];
        float[] query = new float[_qWidth];
        float[] key = new float[_kvWidth];
        float[] value = new float[_kvWidth];
        float[] keys = new float[checked(count * _kvWidth)];
        float[] values = new float[keys.Length];
        double[] scores = new double[count];
        float[] context = new float[_qWidth];
        float[] projected = new float[_embedding];
        float[] gate = new float[_feedForward];
        float[] up = new float[_feedForward];
        float[] activated = new float[_feedForward];

        for (int layer = 0; layer < _shape.Layers; layer++)
        {
            var names = _layers[layer];
            for (int position = 0; position < count; position++)
            {
                Span<float> residual = hidden.AsSpan(position * _embedding, _embedding);

                TransformerOps.RmsNorm(residual, _weights.Vector(names.AttnNorm), _epsilon, normed);
                observe?.Invoke(layer, position, "attn_norm", normed);

                _weights.Project(names.Query, normed, query);
                _weights.Project(names.Key, normed, key);
                _weights.Project(names.Value, normed, value);
                if (names.HasBias)
                {
                    TransformerOps.Add(query, _weights.Vector(names.QueryBias), query);
                    TransformerOps.Add(key, _weights.Vector(names.KeyBias), key);
                    TransformerOps.Add(value, _weights.Vector(names.ValueBias), value);
                }
                if (names.HasHeadNorm)
                {
                    TransformerOps.RmsNormHeads(query, _weights.Vector(names.QueryNorm), _epsilon, _headDim);
                    TransformerOps.RmsNormHeads(key, _weights.Vector(names.KeyNorm), _epsilon, _headDim);
                }
                TransformerOps.RopeHalf(query, _headDim, position, _ropeBase, _headDim);
                TransformerOps.RopeHalf(key, _headDim, position, _ropeBase, _headDim);
                observe?.Invoke(layer, position, "q", query);

                key.CopyTo(keys.AsSpan(position * _kvWidth));
                value.CopyTo(values.AsSpan(position * _kvWidth));
                TransformerOps.GroupedAttention(query, keys.AsSpan(0, (position + 1) * _kvWidth),
                    values.AsSpan(0, (position + 1) * _kvWidth), _shape.HeadCount, _shape.HeadCountKv, _headDim,
                    position + 1, scores, context);
                observe?.Invoke(layer, position, "attn_ctx", context);

                _weights.Project(names.Output, context, projected);
                TransformerOps.Add(residual, projected, residual);
                observe?.Invoke(layer, position, "attn_residual", residual);

                TransformerOps.RmsNorm(residual, _weights.Vector(names.FfnNorm), _epsilon, normed);
                _weights.Project(names.Gate, normed, gate);
                _weights.Project(names.Up, normed, up);
                TransformerOps.SwiGlu(gate, up, activated);
                observe?.Invoke(layer, position, "ffn_act", activated);
                _weights.Project(names.Down, activated, projected);
                TransformerOps.Add(residual, projected, residual);
                observe?.Invoke(layer, position, "out", residual);
            }
        }

        Span<float> last = hidden.AsSpan((count - 1) * _embedding, _embedding);
        TransformerOps.RmsNorm(last, _weights.Vector(_outputNorm), _epsilon, normed);
        observe?.Invoke(-1, -1, "final_norm", normed);

        float[] logits = new float[_shape.VocabularySize];
        _weights.Project(_output, normed, logits);
        observe?.Invoke(-1, -1, "logits", logits);
        return logits;
    }

    private readonly struct LayerNames
    {
        public LayerNames(int layer, IModelWeights weights)
        {
            string prefix = $"blk.{layer}.";
            AttnNorm = prefix + "attn_norm.weight";
            Query = prefix + "attn_q.weight";
            Key = prefix + "attn_k.weight";
            Value = prefix + "attn_v.weight";
            QueryBias = prefix + "attn_q.bias";
            KeyBias = prefix + "attn_k.bias";
            ValueBias = prefix + "attn_v.bias";
            QueryNorm = prefix + "attn_q_norm.weight";
            KeyNorm = prefix + "attn_k_norm.weight";
            Output = prefix + "attn_output.weight";
            FfnNorm = prefix + "ffn_norm.weight";
            Gate = prefix + "ffn_gate.weight";
            Up = prefix + "ffn_up.weight";
            Down = prefix + "ffn_down.weight";
            HasBias = weights.Has(QueryBias);
            HasHeadNorm = weights.Has(QueryNorm);
        }

        public string AttnNorm { get; }
        public string Query { get; }
        public string Key { get; }
        public string Value { get; }
        public string QueryBias { get; }
        public string KeyBias { get; }
        public string ValueBias { get; }
        public string QueryNorm { get; }
        public string KeyNorm { get; }
        public string Output { get; }
        public string FfnNorm { get; }
        public string Gate { get; }
        public string Up { get; }
        public string Down { get; }
        public bool HasBias { get; }
        public bool HasHeadNorm { get; }
    }
}

namespace Satori.Core.Model;

/// <summary>
/// Incremental dense session for the qwen2/qwen3 families. It owns a key/value state sized from the model
/// shape and a chosen context length, and reuses the shared per-layer step of <see cref="DenseForward"/>,
/// so incremental execution and full recomputation agree by construction.
/// </summary>
/// <remarks>
/// State is one F32 key and one F32 value per KV head and head dimension for every layer and position:
/// <c>Layers * 2 * HeadCountKv * HeadDim * ContextLength * 4</c> bytes, matching <see cref="ModelPlanner"/>.
/// Only the incremental step is stored; the prompt and each decoded token reuse it. The session is not
/// thread-safe and is usable until <see cref="Dispose"/>.
/// </remarks>
public sealed class DenseSession : IInferenceSession
{
    private readonly ModelDefinition _model;
    private readonly DenseForward _forward;
    private readonly ModelShape _shape;
    private readonly int _context;
    private readonly int _kvWidth;
    private readonly float[] _keys;
    private readonly float[] _values;
    private readonly float[] _hidden;
    private readonly float[] _logits;
    private readonly DenseScratch _scratch;
    private readonly ForwardObserver? _observe;
    private int _position;
    private bool _disposed;

    /// <summary>
    /// Creates a session. <paramref name="contextLength"/> bounds the number of positions; the key/value
    /// state is dimensioned from the model shape and this bound, never from a nominal constant.
    /// </summary>
    public DenseSession(ModelDefinition model, IModelWeights weights, int contextLength, ForwardObserver? observe = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        if (model.Shape is null)
            throw new InvalidOperationException("The definition has no derived shape; it is not executable.");
        if (!model.Capabilities.IsExecutable)
            throw new NotSupportedException("The definition has capability gaps and cannot be executed.");

        _model = model;
        _shape = model.Shape;
        _context = contextLength;
        _kvWidth = _shape.KvWidth;
        _observe = observe;
        _forward = new DenseForward(_shape, weights);

        long slots = checked((long)_shape.Layers * 2 * _context * _kvWidth);
        if (slots > int.MaxValue)
            throw new NotSupportedException($"Key/value state of {slots} floats exceeds Int32.");
        _keys = new float[(int)slots / 2];
        _values = new float[(int)slots / 2];
        _hidden = new float[_shape.EmbeddingLength];
        _logits = new float[_shape.VocabularySize];
        _scratch = _forward.CreateScratch(_context);
    }

    public ModelDefinition Model => _model;

    public int ContextLength => _context;

    public int Position => _position;

    /// <summary>Bytes owned by the key/value state, consistent with <see cref="ModelPlan.StateBytes"/>.</summary>
    public long StateBytes => checked((long)_shape.Layers * 2 * _context * _kvWidth * ModelPlanner.F32Bytes);

    /// <summary>Logits of the most recently appended position.</summary>
    public ReadOnlySpan<float> Logits => _logits;

    /// <summary>Appends a single token and returns the logits of that position.</summary>
    public ReadOnlySpan<float> Append(int token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)token >= (uint)_shape.VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(token), $"Token {token} is outside the vocabulary.");
        if (_position >= _context)
            throw new InvalidOperationException($"Context limit {_context} reached; reset or enlarge the session.");
        _forward.Embed(token, _hidden);
        for (int layer = 0; layer < _shape.Layers; layer++)
        {
            Span<float> keys = _keys.AsSpan(layer * _context * _kvWidth, _context * _kvWidth);
            Span<float> values = _values.AsSpan(layer * _context * _kvWidth, _context * _kvWidth);
            _forward.Step(layer, _position, _hidden, keys, values, _scratch, _observe);
        }
        _forward.Final(_hidden, _logits, _scratch, _observe);
        _position++;
        return _logits;
    }

    public void Forward(ReadOnlySpan<int> tokens, Span<float> logits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (logits.Length != _shape.VocabularySize)
            throw new ArgumentException("Logits buffer must match the vocabulary size.", nameof(logits));
        if (_position + tokens.Length > _context)
            throw new ArgumentOutOfRangeException(nameof(tokens),
                $"Appending {tokens.Length} tokens at position {_position} exceeds the context limit {_context}.");
        foreach (int token in tokens) Append(token);
        _logits.CopyTo(logits);
    }

    public void Reset() => _position = 0;

    public void Dispose()
    {
        _disposed = true;
        _position = 0;
    }
}

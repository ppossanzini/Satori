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
/// <para>
/// <see cref="Forward"/> runs the prompt as a batched prefill: it walks the prompt in chunks and hands each
/// chunk to <see cref="DenseForward.LayerBatch"/>, so every projection decodes each weight block once for the
/// whole chunk instead of once per position. <see cref="Append"/> stays the single-position step used for
/// decoding, and the two agree because they run the same operator sequence from the same buffers.
/// </para>
/// </remarks>
public sealed class DenseSession : IInferenceSession
{
    /// <summary>Positions one prefill chunk holds; large enough that each weight block pays for its own decode.</summary>
    private const int ChunkPositions = DenseBatchScratch.MaxPositions;

    private readonly ModelDefinition _model;
    private readonly DenseForward _forward;
    private readonly ModelShape _shape;
    private readonly int _context;
    private readonly int _kvWidth;
    private readonly int _embedding;
    private readonly InferenceOptions _options;
    private readonly float[] _keys;
    private readonly float[] _values;
    private readonly float[] _hidden;
    private readonly int _chunkPositions;
    private readonly float[] _logits;
    private readonly DenseScratch _scratch;
    private readonly DenseBatchScratch _batch;
    private readonly ForwardObserver? _observe;
    private int _position;
    private bool _disposed;

    /// <summary>
    /// Creates a session. <paramref name="contextLength"/> bounds the number of positions; the key/value
    /// state is dimensioned from the model shape and this bound, never from a nominal constant.
    /// </summary>
    public DenseSession(ModelDefinition model, IModelWeights weights, int contextLength,
        ForwardObserver? observe = null, InferenceOptions? options = null)
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
        _embedding = _shape.EmbeddingLength;
        _observe = observe;
        _options = (options ?? InferenceOptions.Default).Validated();
        _forward = new DenseForward(_shape, weights);

        // The budget is applied here, before any large allocation, because a session that cannot fit must be
        // refused rather than allowed to commit the memory and fail later on an allocation it cannot honour.
        // The plan counts the compressed weights, the key/value state for this context and the per-layer
        // workspace, which is the same account the operator sees from `describe`.
        if (_options.BudgetBytes > 0)
        {
            ModelPlan plan = model.Plan(contextLength, _options.BudgetBytes);
            if (plan.ExceedsBudget)
                throw new InvalidOperationException(
                    $"The session needs {plan.TotalBytes} bytes and the budget is {plan.BudgetBytes}: " +
                    $"weights {plan.WeightBytes}, state {plan.StateBytes}, workspace {plan.WorkspaceBytes}.");
        }

        long slots = checked((long)_shape.Layers * 2 * _context * _kvWidth);
        if (slots > int.MaxValue)
            throw new NotSupportedException($"Key/value state of {slots} floats exceeds Int32.");
        _keys = new float[(int)slots / 2];
        _values = new float[(int)slots / 2];
        _hidden = new float[_shape.EmbeddingLength];
        _logits = new float[_shape.VocabularySize];
        _chunkPositions = Math.Min(_context, ChunkPositions);
        _scratch = _forward.CreateScratch(_context);
        _batch = new DenseBatchScratch(_shape, _chunkPositions, _context);
    }

    public ModelDefinition Model => _model;

    public int ContextLength => _context;

    public int Position => _position;

    /// <summary>Bytes owned by the key/value state, consistent with <see cref="ModelPlan.StateBytes"/>.</summary>
    public long StateBytes => checked((long)_shape.Layers * 2 * _context * _kvWidth * ModelPlanner.F32Bytes);

    /// <summary>Budget this session was created with, or zero when the host stated none.</summary>
    public long BudgetBytes => _options.BudgetBytes;

    /// <summary>Logits of the most recently appended position.</summary>
    public ReadOnlySpan<float> Logits => _logits;

    /// <summary>Appends a single token and returns the logits of that position.</summary>
    public ReadOnlySpan<float> Append(int token) => Append(token, CancellationToken.None);

    /// <summary>Appends a single token, observing <paramref name="cancellationToken"/> before every projection.</summary>
    public ReadOnlySpan<float> Append(int token, CancellationToken cancellationToken)
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
            _forward.Step(layer, _position, _hidden, keys, values, _scratch, _observe, _options.Threads, cancellationToken);
        }
        _forward.Final(_hidden, _logits, _scratch, _observe, cancellationToken: cancellationToken);
        _position++;
        // A request that arrived during the last projection is honored before this call reports success, so a
        // cancelled call never looks like a completed one. The position it advanced by is work that was done.
        cancellationToken.ThrowIfCancellationRequested();
        return _logits;
    }

    public void Forward(ReadOnlySpan<int> tokens, Span<float> logits) =>
        Forward(tokens, logits, CancellationToken.None);

    public void Forward(ReadOnlySpan<int> tokens, Span<float> logits, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (tokens.IsEmpty) throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (logits.Length != _shape.VocabularySize)
            throw new ArgumentException("Logits buffer must match the vocabulary size.", nameof(logits));
        if (_position + tokens.Length > _context)
            throw new ArgumentOutOfRangeException(nameof(tokens),
                $"Appending {tokens.Length} tokens at position {_position} exceeds the context limit {_context}.");

        // A chunk commits its positions only once every layer of it has run, so a stop request between chunks
        // leaves the session at a chunk boundary rather than mid-prompt. Within a chunk the checkpoints inside
        // each layer bound the delay.
        int lastCount = 0;
        for (int done = 0; done < tokens.Length; done += _chunkPositions)
        {
            cancellationToken.ThrowIfCancellationRequested();
            lastCount = Math.Min(_chunkPositions, tokens.Length - done);
            PrefillChunk(tokens.Slice(done, lastCount), cancellationToken);
        }

        // Prefill does not need the logits of the positions a real continuation will never resample from, so
        // the output projection runs once for the last prompt position instead of once for every one of them.
        _forward.Final(Hidden(lastCount - 1), _logits, _scratch, _observe, cancellationToken: cancellationToken);
        _logits.CopyTo(logits);
        // Same as Append: a request raised during the last projection must not be reported as a completed run.
        cancellationToken.ThrowIfCancellationRequested();
    }

    /// <summary>
    /// Runs one chunk of positions through every layer by calling the batched layer once per layer. The chunk
    /// holds positions <c>start..start+count-1</c>; the layer writes each position's key and value at its
    /// absolute slot, and the next chunk appends after them.
    /// </summary>
    private void PrefillChunk(ReadOnlySpan<int> tokens, CancellationToken cancellationToken)
    {
        int start = _position;
        int count = tokens.Length;
        for (int i = 0; i < count; i++) _forward.Embed(tokens[i], Hidden(i));

        for (int layer = 0; layer < _shape.Layers; layer++)
            _forward.LayerBatch(layer, start, count, _batch, LayerKeys(layer), LayerValues(layer),
                _options.Threads, _observe, cancellationToken);

        _position = start + count;
    }

    private Span<float> Hidden(int index) => _batch.Hidden.AsSpan(index * _embedding, _embedding);

    private Span<float> LayerKeys(int layer) => _keys.AsSpan(layer * _context * _kvWidth, _context * _kvWidth);

    private Span<float> LayerValues(int layer) => _values.AsSpan(layer * _context * _kvWidth, _context * _kvWidth);

    public void Reset() => _position = 0;

    public void Dispose()
    {
        _disposed = true;
        _position = 0;
    }
}

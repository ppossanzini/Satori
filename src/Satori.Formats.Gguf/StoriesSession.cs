namespace Satori.Formats.Gguf;

/// <summary>Single-sequence FP32 KV cache borrowing the model's lifetime.</summary>
public sealed class StoriesSession : IDisposable
{
    public const int CacheByteLength = 2*6*StoriesModel.ContextLength*StoriesModel.Width*sizeof(float);
    private readonly StoriesModel _model;
    private readonly bool _scalar;
    private readonly object _sync = new();
    private StoriesWorkspace? _workspace;
    private readonly IStoriesSessionExecution? _execution;
    private readonly float[]? _embedding;
    private bool _disposed;
    private int _count;
    private bool _failed;

    internal StoriesSession(StoriesModel model, bool scalar)
    {
        _model = model;
        _scalar = scalar;
        _execution=model.CreateExecution(scalar);
        if(_execution is null)_workspace=new();else _embedding=new float[StoriesModel.Width];
    }

    public int Count { get { lock (_sync) { ObjectDisposedException.ThrowIf(_disposed,this); return _count; } } }

    /// <summary>Writes logits after appending one token. Returns false at the context limit,
    /// leaving both logits and cache unchanged. No allocations after warm-up.
    /// Execution failure clears the cache and requires Reset before another append.</summary>
    public bool TryAppend(int token, Span<float> logits)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            if (_failed) throw new InvalidOperationException("Session failed; reset before reusing it.");
            if ((uint)token >= StoriesModel.VocabularySize) throw new ArgumentOutOfRangeException(nameof(token));
            if (logits.Length != StoriesModel.VocabularySize) throw new ArgumentException("Expected 32000 logits.",nameof(logits));
            if (_count == StoriesModel.ContextLength) return false;
            try
            {
                if(_execution is null)_model.ComputeNext(_workspace!,token,_count,logits,_scalar);
                else _model.ComputeResidentNext(_execution,_embedding!,token,_count,logits);
            }
            catch
            {
                _workspace?.Clear();
                _failed = true;
                _count = 0;
                throw;
            }
            _count++;
            return true;
        }
    }

    public void Reset()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            _execution?.Reset();
            _workspace?.Clear();
            _failed = false;
            _count = 0;
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _execution?.Dispose();
            _disposed=true;
            _workspace?.Clear();
            _workspace = null;
            _count = 0;
        }
    }
}

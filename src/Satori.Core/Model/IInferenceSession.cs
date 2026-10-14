namespace Satori.Core.Model;

/// <summary>
/// A per-context incremental inference session. It appends tokens to its state and returns the logits of the
/// last appended position. Calling it past <see cref="ContextLength"/> is an error, never a silent
/// truncation. Sessions are private and not thread-safe: one <see cref="Forward"/> at a time.
/// </summary>
public interface IInferenceSession : IDisposable
{
    /// <summary>The immutable definition this session executes.</summary>
    ModelDefinition Model { get; }

    /// <summary>Maximum number of positions this session accepts.</summary>
    int ContextLength { get; }

    /// <summary>Number of committed positions.</summary>
    int Position { get; }

    /// <summary>Discards all committed positions; the session returns to its initial state.</summary>
    void Reset();

    /// <summary>
    /// Appends tokens, advances <see cref="Position"/> and writes the logits of the last position.
    /// <paramref name="cancellationToken"/> is observed before every projection, so a request to stop is
    /// honored within one projection of the largest shape in the model and never leaves a partially written
    /// position committed.
    /// </summary>
    void Forward(ReadOnlySpan<int> tokens, Span<float> logits, CancellationToken cancellationToken = default);
}

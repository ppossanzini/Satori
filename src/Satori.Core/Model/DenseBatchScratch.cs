namespace Satori.Core.Model;

/// <summary>
/// Activation buffers for a batched prefill of <paramref name="positions"/> positions. One slice per
/// position holds what a per-position step would have held in a <see cref="DenseScratch"/>, which is what
/// lets a projection see every position of a chunk at once and decode each weight block only once.
/// </summary>
/// <remarks>
/// Sized once from the shape and the chunk, never from a nominal constant. The activation buffers are
/// transient: only the key/value state survives a chunk, so the memory here is reclaimed after the prefill.
/// </remarks>
internal sealed class DenseBatchScratch
{
    internal const int MaxPositions = 256;

    public DenseBatchScratch(ModelShape shape, int positions, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(positions);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokens);
        Hidden = new float[checked(positions * shape.EmbeddingLength)];
        Normed = new float[checked(positions * shape.EmbeddingLength)];
        Queries = new float[checked(positions * shape.QueryWidth)];
        Keys = new float[checked(positions * shape.KvWidth)];
        Values = new float[checked(positions * shape.KvWidth)];
        Context = new float[checked(positions * shape.QueryWidth)];
        Projected = new float[checked(positions * shape.EmbeddingLength)];
        Gate = new float[checked(positions * shape.FeedForwardLength)];
        Up = new float[checked(positions * shape.FeedForwardLength)];
        Activated = new float[checked(positions * shape.FeedForwardLength)];
        Down = new float[checked(positions * shape.EmbeddingLength)];
        Scores = new double[maxTokens];
    }

    public float[] Hidden { get; }
    public float[] Normed { get; }
    public float[] Queries { get; }
    public float[] Keys { get; }
    public float[] Values { get; }
    public float[] Context { get; }
    public float[] Projected { get; }
    public float[] Gate { get; }
    public float[] Up { get; }
    public float[] Activated { get; }
    public float[] Down { get; }

    /// <summary>Scratch for one position's attention scores; attention runs one position at a time.</summary>
    public double[] Scores { get; }
}

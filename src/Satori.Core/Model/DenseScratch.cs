namespace Satori.Core.Model;

/// <summary>
/// Reusable per-position activation scratch for one dense layer step. Sized once from the model shape; it
/// does not grow with the number of tokens beyond the attention score array, which covers
/// <paramref name="maxTokens"/> positions. Owned by exactly one session or forward pass at a time.
/// </summary>
internal sealed class DenseScratch
{
    public DenseScratch(ModelShape shape, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokens);
        Normed = new float[shape.EmbeddingLength];
        Query = new float[shape.QueryWidth];
        Key = new float[shape.KvWidth];
        Value = new float[shape.KvWidth];
        Context = new float[shape.QueryWidth];
        Projected = new float[shape.EmbeddingLength];
        Gate = new float[shape.FeedForwardLength];
        Up = new float[shape.FeedForwardLength];
        Activated = new float[shape.FeedForwardLength];
        Scores = new double[maxTokens];
    }

    public float[] Normed { get; }
    public float[] Query { get; }
    public float[] Key { get; }
    public float[] Value { get; }
    public float[] Context { get; }
    public float[] Projected { get; }
    public float[] Gate { get; }
    public float[] Up { get; }
    public float[] Activated { get; }
    public double[] Scores { get; }
}

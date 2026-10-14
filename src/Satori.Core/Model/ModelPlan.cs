namespace Satori.Core.Model;

/// <summary>
/// Shape and memory plan computed before generation. Weights are the compressed payload; state is the
/// KV/recurrent store for <see cref="ContextLength"/> positions; workspace is the per-layer scratch upper
/// bound. All arithmetic is checked.
/// </summary>
public sealed record ModelPlan(
    int ContextLength,
    long WeightBytes,
    long StateBytes,
    long WorkspaceBytes,
    long TotalBytes,
    long BudgetBytes)
{
    public bool ExceedsBudget => BudgetBytes > 0 && TotalBytes > BudgetBytes;
}

/// <summary>Derives a plan from a validated shape. Pure function so plans can be tested without a container.</summary>
public static class ModelPlanner
{
    public const int F32Bytes = 4;

    /// <summary>
    /// State is one F32 key and one F32 value per KV head and head dimension for every layer and position.
    /// Workspace covers the single-position and bounded batched activation buffers, logits, both attention
    /// score buffers and pooled Q8_K activations. It excludes runtime and host-process overhead.
    /// </summary>
    public static ModelPlan Plan(ModelShape shape, long weightBytes, int contextLength, long budgetBytes)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        ArgumentOutOfRangeException.ThrowIfNegative(weightBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(budgetBytes);

        long state = checked((long)shape.Layers * 2 * shape.HeadCountKv * shape.HeadDim * contextLength * F32Bytes);
        int positions = Math.Min(contextLength, DenseBatchScratch.MaxPositions);
        long single = checked((long)shape.EmbeddingLength * 3 + (long)shape.QueryWidth * 2
            + (long)shape.KvWidth * 2 + (long)shape.FeedForwardLength * 3 + shape.VocabularySize);
        long batch = checked(((long)shape.EmbeddingLength * 4 + (long)shape.QueryWidth * 2
            + (long)shape.KvWidth * 2 + (long)shape.FeedForwardLength * 3) * positions);
        long activationColumns = Math.Max(shape.EmbeddingLength, shape.FeedForwardLength);
        long activationBlocks = (activationColumns + Q8KCodec.ElementsPerBlock - 1) / Q8KCodec.ElementsPerBlock;
        long activationBytes = checked(positions * activationBlocks * Q8KCodec.BytesPerBlock);
        long pooledBytes = 16;
        while (pooledBytes < activationBytes) pooledBytes = checked(pooledBytes * 2);
        long workspace = checked((single + batch) * F32Bytes + (long)contextLength * sizeof(double) * 2
            + pooledBytes * 2 + 4096);
        long total = checked(checked(weightBytes + state) + workspace);
        return new ModelPlan(contextLength, weightBytes, state, workspace, total, budgetBytes);
    }
}

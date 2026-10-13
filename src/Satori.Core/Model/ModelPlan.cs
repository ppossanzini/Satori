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
    /// Workspace is a documented upper bound on per-layer activation scratch plus the attention score buffer
    /// for the full context; it does not grow with the number of stored positions beyond that buffer.
    /// </summary>
    public static ModelPlan Plan(ModelShape shape, long weightBytes, int contextLength, long budgetBytes)
    {
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(contextLength);
        ArgumentOutOfRangeException.ThrowIfNegative(weightBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(budgetBytes);

        long state = checked((long)shape.Layers * 2 * shape.HeadCountKv * shape.HeadDim * contextLength * F32Bytes);
        long workspace = checked(((long)shape.EmbeddingLength * 4
            + (long)shape.FeedForwardLength * 2
            + (long)shape.HeadCount * contextLength) * F32Bytes);
        long total = checked(checked(weightBytes + state) + workspace);
        return new ModelPlan(contextLength, weightBytes, state, workspace, total, budgetBytes);
    }
}

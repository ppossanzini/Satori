namespace Satori.Core.Model;

/// <summary>
/// How much of the machine one inference session may use. Declared once by the host and immutable for the
/// session's lifetime, so the kernels never read a process-wide setting.
/// </summary>
/// <remarks>
/// <see cref="Threads"/> bounds the degree of parallelism of every multi-position phase; it never changes a
/// result, because phases split only units whose outputs are disjoint.
/// </remarks>
public sealed record InferenceOptions(int Threads)
{
    /// <summary>The protocol's inference thread budget, used when the host states no limit.</summary>
    public const int BudgetThreads = 8;

    /// <summary>The budget thread count, clamped to the machine.</summary>
    public static InferenceOptions Default { get; } = new(Math.Min(BudgetThreads, Environment.ProcessorCount));

    /// <summary>Rejects a thread count that cannot be served.</summary>
    public InferenceOptions Validated()
    {
        if (Threads < 1) throw new ArgumentOutOfRangeException(nameof(Threads), "Threads must be at least one.");
        return this;
    }
}

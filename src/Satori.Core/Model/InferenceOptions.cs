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

    /// <summary>The budget thread count, clamped to the machine's logical processors.</summary>
    /// <remarks>
    /// Clamping to physical cores instead was implemented and measured, and it was not kept. On a hybrid host
    /// the budget fell from eight to six when the process was confined to the six performance cores, and the
    /// two Q6_K tensors of the real artifact ran 1.06x faster at best under a within-configuration spread of
    /// 1.64x: below the noise, and therefore unresolved rather than a win. What the earlier session attributed
    /// to the extra workers sharing cores through simultaneous multithreading was mostly the confinement
    /// itself, since <c>taskset</c> restricts the whole process — pool, JIT and collector included — and not
    /// only the workers. See docs/benchmarks/thread-budget-2026-10-14/.
    /// </remarks>
    public static InferenceOptions Default { get; } = new(Math.Min(BudgetThreads, Environment.ProcessorCount));

    /// <summary>Rejects a thread count that cannot be served.</summary>
    public InferenceOptions Validated()
    {
        if (Threads < 1) throw new ArgumentOutOfRangeException(nameof(Threads), "Threads must be at least one.");
        return this;
    }
}

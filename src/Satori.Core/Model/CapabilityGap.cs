namespace Satori.Core.Model;

/// <summary>A single missing capability, with the reason it blocks execution.</summary>
/// <param name="Kind">One of <c>architecture</c>, <c>tensor-format</c>, <c>operator</c>, <c>tokenizer</c>,
/// <c>transformation</c>, <c>shape</c>, <c>budget</c>, <c>backend</c>.</param>
/// <param name="Detail">The specific item, for example a tensor type id or a metadata key.</param>
/// <param name="Reason">Why it is missing.</param>
public sealed record CapabilityGap(string Kind, string Detail, string Reason);

/// <summary>Result of checking a definition against the running build.</summary>
public sealed class CapabilityReport
{
    public CapabilityReport(IReadOnlyList<CapabilityGap> gaps) => Gaps = gaps;

    public IReadOnlyList<CapabilityGap> Gaps { get; }
    public bool IsExecutable => Gaps.Count == 0;

    public override string ToString() =>
        IsExecutable ? "executable" : string.Join("; ", Gaps.Select(g => $"{g.Kind}:{g.Detail}:{g.Reason}"));
}

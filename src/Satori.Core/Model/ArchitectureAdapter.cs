namespace Satori.Core.Model;

/// <summary>
/// Resolves a validated <see cref="ModelShape"/> for one architecture family and reports every missing or
/// inconsistent requirement. Adapters live in the capability registry; core interfaces never carry a model
/// name.
/// </summary>
public interface IArchitectureAdapter
{
    IReadOnlyList<string> ArchitectureIds { get; }

    /// <summary>
    /// Validates metadata and tensor shapes for the family. Returns the gaps that block execution and, when
    /// the shape is derivable, the derived <see cref="ModelShape"/>.
    /// </summary>
    IReadOnlyList<CapabilityGap> Validate(
        ModelConfig config, IReadOnlyDictionary<string, ModelTensor> tensors, out ModelShape? shape);
}

/// <summary>Capability registry of architecture adapters. Unknown architectures resolve to <c>null</c>.</summary>
public static class ArchitectureAdapters
{
    private static readonly IArchitectureAdapter[] All = [new DenseTransformerAdapter()];

    public static IArchitectureAdapter? Resolve(string architectureId) =>
        All.FirstOrDefault(adapter => adapter.ArchitectureIds.Contains(architectureId, StringComparer.Ordinal));

    public static IReadOnlyList<string> KnownIds { get; } =
        All.SelectMany(adapter => adapter.ArchitectureIds).ToArray();
}

namespace Satori.Core.Model;

/// <summary>
/// Immutable, validated description of a model. It carries descriptors only, no weights, and is safe to
/// share between sessions. <see cref="Capabilities"/> is empty when the definition is executable.
/// </summary>
public sealed class ModelDefinition
{
    public ModelDefinition(
        string architecture,
        ModelConfig config,
        IReadOnlyList<ModelTensor> tensors,
        IReadOnlyList<CapabilityGap> gaps,
        ModelShape? shape)
    {
        Architecture = architecture;
        Config = config;
        Tensors = tensors;
        Capabilities = new CapabilityReport(gaps);
        Shape = shape;
        var map = new Dictionary<string, ModelTensor>(tensors.Count, StringComparer.Ordinal);
        long weightBytes = 0;
        foreach (var tensor in tensors)
        {
            if (tensor.ByteLengthKnown) weightBytes += tensor.PayloadBytes;
            map[tensor.Name] = tensor;
        }
        ByName = map;
        WeightBytes = weightBytes;
    }

    public string Architecture { get; }
    public ModelConfig Config { get; }
    public IReadOnlyList<ModelTensor> Tensors { get; }
    public IReadOnlyDictionary<string, ModelTensor> ByName { get; }
    public CapabilityReport Capabilities { get; }
    public ModelShape? Shape { get; }
    public long WeightBytes { get; }

    public bool TryGetTensor(string name, out ModelTensor tensor) => ByName.TryGetValue(name, out tensor!);

    /// <summary>
    /// Computes the shape/memory plan. Requires only a derived shape; callers check
    /// <see cref="Capabilities"/> before generation.
    /// </summary>
    public ModelPlan Plan(int contextLength, long budgetBytes)
    {
        if (Shape is null)
            throw new InvalidOperationException("Model shape was not derived; the definition is not shape-valid.");
        return ModelPlanner.Plan(Shape, WeightBytes, contextLength, budgetBytes);
    }
}

using Satori.Core.Model;

namespace Satori.Formats.Gguf;

/// <summary>
/// Builds a validated <see cref="ModelDefinition"/> from a GGUF container. It uses the tolerant descriptor
/// inventory so that files with unsupported tensor types or unknown architectures still produce an explicit
/// capability report instead of failing to open. Tensor payloads are not read.
/// </summary>
public static class GgufModelLoader
{
    public static ModelDefinition Load(string path)
    {
        var inventory = GgufInventory.Scan(path);

        var values = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var entry in inventory.Metadata)
            if (entry.Type != 9 && entry.Value is not null)
                values[entry.Key] = entry.Value;
        var config = new ModelConfig(inventory.Architecture, values);

        var gaps = new List<CapabilityGap>();
        var tensors = new List<ModelTensor>(inventory.Tensors.Count);
        var unsupported = new List<string>();
        var unsupportedSeen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var descriptor in inventory.Tensors)
        {
            if (!descriptor.Supported && unsupportedSeen.Add($"{descriptor.TypeName}({descriptor.TypeId})"))
                unsupported.Add($"{descriptor.TypeName}({descriptor.TypeId})");

            var dimensions = new int[descriptor.Dimensions.Count];
            bool overflow = false;
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (descriptor.Dimensions[i] > int.MaxValue) { overflow = true; break; }
                dimensions[i] = (int)descriptor.Dimensions[i];
            }
            if (overflow)
            {
                gaps.Add(new CapabilityGap("shape", descriptor.Name, "declared dimension exceeds Int32"));
                continue;
            }
            tensors.Add(new ModelTensor(descriptor.Name, (int)descriptor.TypeId, descriptor.TypeName,
                descriptor.Supported, dimensions, descriptor.Offset, descriptor.ByteLength, descriptor.LengthKnown));
        }
        foreach (string type in unsupported)
            gaps.Add(new CapabilityGap("tensor-format", type, "tensor type is not decodable by the current profile"));

        var tensorMap = tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        var adapter = ArchitectureAdapters.Resolve(inventory.Architecture);
        ModelShape? shape = null;
        if (adapter is null)
            gaps.Add(new CapabilityGap("architecture", inventory.Architecture, "no adapter is registered for this architecture"));
        else
            gaps.AddRange(adapter.Validate(config, tensorMap, out shape));

        return new ModelDefinition(inventory.Architecture, config, tensors, gaps, shape);
    }
}

namespace Satori.Formats.Gguf;

internal sealed class StoriesLayer(GgufFile file, int index)
{
    internal GgufFloatVector AttentionNorm { get; } = new(file, $"blk.{index}.attn_norm.weight");
    internal GgufFloatVector FeedForwardNorm { get; } = new(file, $"blk.{index}.ffn_norm.weight");
    internal GgufQuantizedMatrix Query { get; } = new(file, $"blk.{index}.attn_q.weight");
    internal GgufQuantizedMatrix Key { get; } = new(file, $"blk.{index}.attn_k.weight");
    internal GgufQuantizedMatrix Value { get; } = new(file, $"blk.{index}.attn_v.weight");
    internal GgufQuantizedMatrix AttentionOutput { get; } = new(file, $"blk.{index}.attn_output.weight");
    internal GgufQuantizedMatrix Gate { get; } = new(file, $"blk.{index}.ffn_gate.weight");
    internal GgufQuantizedMatrix Up { get; } = new(file, $"blk.{index}.ffn_up.weight");
    internal GgufQuantizedMatrix Down { get; } = new(file, $"blk.{index}.ffn_down.weight");
    internal string OutputState { get; } = $"l_out-{index}";
}

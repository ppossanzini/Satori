using Satori.Core.Model;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS05 critical tests: shape incompatibility, arithmetic overflow and missing capabilities.
/// </summary>
public sealed class ModelDescriptorTests
{
    [Fact]
    public void DenseAdapterDerivesShapeAndReportsNoGaps()
    {
        var (config, tensors) = Qwen2(embedding: 8, heads: 2, headsKv: 1, ffn: 16, vocabulary: 10, layers: 2);
        var gaps = new DenseTransformerAdapter().Validate(config, tensors, out var shape);

        Assert.Empty(gaps);
        Assert.NotNull(shape);
        Assert.Equal(2, shape!.Layers);
        Assert.Equal(8, shape.EmbeddingLength);
        Assert.Equal(4, shape.HeadDim);          // 8 / 2, no explicit key_length
        Assert.Equal(1, shape.HeadCountKv);
    }

    [Fact]
    public void ExplicitKeyLengthOverridesHiddenOverHeads()
    {
        // Embedding 8 with 2 heads would give head_dim 4, but the metadata declares 6 -> Q width 12.
        var (config, tensors) = Qwen2(embedding: 8, heads: 2, headsKv: 1, ffn: 16, vocabulary: 10, layers: 1,
            keyLength: 6);
        var gaps = new DenseTransformerAdapter().Validate(config, tensors, out var shape);

        Assert.Empty(gaps);
        Assert.Equal(6, shape!.HeadDim);
    }

    [Fact]
    public void IncompatibleShapeIsReported()
    {
        var (config, tensors) = Qwen2(embedding: 8, heads: 2, headsKv: 1, ffn: 16, vocabulary: 10, layers: 1);
        tensors["blk.0.attn_q.weight"] = Tensor("blk.0.attn_q.weight", [8, 999]); // wrong output width

        var gaps = new DenseTransformerAdapter().Validate(config, tensors, out _);

        var gap = Assert.Single(gaps);
        Assert.Equal("shape", gap.Kind);
        Assert.Equal("blk.0.attn_q.weight", gap.Detail);
    }

    [Fact]
    public void UndeclaredHeadLengthWithIndivisibleEmbeddingIsReported()
    {
        var (config, tensors) = Qwen2(embedding: 10, heads: 4, headsKv: 1, ffn: 16, vocabulary: 10, layers: 1);
        var gaps = new DenseTransformerAdapter().Validate(config, tensors, out _);
        Assert.Contains(gaps, g => g.Kind == "architecture" && g.Detail == "attention.key_length");
    }

    [Fact]
    public void MissingRequiredMetadataIsReported()
    {
        var values = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["qwen2.block_count"] = 1L, // embedding_length and the rest are absent
        };
        var gaps = new DenseTransformerAdapter().Validate(new ModelConfig("qwen2", values),
            new Dictionary<string, ModelTensor>(), out var shape);

        Assert.Null(shape);
        Assert.Contains(gaps, g => g.Detail == "embedding_length");
    }

    [Fact]
    public void UnknownArchitectureHasNoAdapter()
    {
        Assert.Null(ArchitectureAdapters.Resolve("mystery-moe"));
        Assert.NotNull(ArchitectureAdapters.Resolve("qwen3"));
    }

    [Fact]
    public void PlanOverflowIsRejected()
    {
        // layers * 2 * kv * headDim * context * 4 overflows Int64 for these declarations.
        var huge = new ModelShape(int.MaxValue, 1, 1, 1, 1, 1, 1, false, 1e-6, 10000);
        Assert.Throws<OverflowException>(() => ModelPlanner.Plan(huge, 1, int.MaxValue, 0));
    }

    [Fact]
    public void PlanSumsWeightsStateAndWorkspace()
    {
        var shape = new ModelShape(2, 8, 16, 2, 1, 4, 10, false, 1e-6, 10000);
        var plan = ModelPlanner.Plan(shape, weightBytes: 100, contextLength: 2048, budgetBytes: 12L << 30);

        Assert.Equal(2L * 2 * 1 * 4 * 2048 * 4, plan.StateBytes);
        Assert.Equal(100 + plan.StateBytes + plan.WorkspaceBytes, plan.TotalBytes);
        Assert.False(plan.ExceedsBudget);

        var tiny = ModelPlanner.Plan(shape, weightBytes: 100, contextLength: 2048, budgetBytes: 1);
        Assert.True(tiny.ExceedsBudget);
    }

    [Fact]
    public void UnsupportedTensorTypeBecomesAFormatGap()
    {
        byte[] bytes = GgufContainerFixture.Prefix(32, tensors:
        [
            ("q4k", [256, 2], 12, 0),
            ("q6k", [256, 1], 14, 288),
        ]);
        using var temp = new MappedFileFixture(bytes.Length, 0, bytes);
        var definition = GgufModelLoader.Load(temp.Path);

        Assert.False(definition.Capabilities.IsExecutable);
        // Q4_K is supported since DW06.1; Q6_K still is not, so it is the format gap here.
        Assert.DoesNotContain(definition.Capabilities.Gaps, g => g.Kind == "tensor-format" && g.Detail == "Q4_K(12)");
        Assert.Contains(definition.Capabilities.Gaps, g => g.Kind == "tensor-format" && g.Detail == "Q6_K(14)");
        Assert.Contains(definition.Capabilities.Gaps, g => g.Kind == "architecture");
        Assert.Null(definition.Shape);
    }

    private static (ModelConfig, Dictionary<string, ModelTensor>) Qwen2(
        int embedding, int heads, int headsKv, int ffn, int vocabulary, int layers, int? keyLength = null)
    {
        int headDim = keyLength ?? embedding / heads;
        var values = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["qwen2.block_count"] = (long)layers,
            ["qwen2.embedding_length"] = (long)embedding,
            ["qwen2.feed_forward_length"] = (long)ffn,
            ["qwen2.attention.head_count"] = (long)heads,
            ["qwen2.attention.head_count_kv"] = (long)headsKv,
            ["qwen2.attention.layer_norm_rms_epsilon"] = 1e-6,
            ["qwen2.rope.freq_base"] = 10000.0,
        };
        if (keyLength is int length) values["qwen2.attention.key_length"] = (long)length;

        var tensors = new Dictionary<string, ModelTensor>(StringComparer.Ordinal)
        {
            ["token_embd.weight"] = Tensor("token_embd.weight", [embedding, vocabulary]),
            ["output.weight"] = Tensor("output.weight", [embedding, vocabulary]),
            ["output_norm.weight"] = Tensor("output_norm.weight", [embedding]),
        };
        for (int layer = 0; layer < layers; layer++)
        {
            string p = $"blk.{layer}.";
            tensors[p + "attn_norm.weight"] = Tensor(p + "attn_norm.weight", [embedding]);
            tensors[p + "attn_q.weight"] = Tensor(p + "attn_q.weight", [embedding, heads * headDim]);
            tensors[p + "attn_k.weight"] = Tensor(p + "attn_k.weight", [embedding, headsKv * headDim]);
            tensors[p + "attn_v.weight"] = Tensor(p + "attn_v.weight", [embedding, headsKv * headDim]);
            tensors[p + "attn_output.weight"] = Tensor(p + "attn_output.weight", [heads * headDim, embedding]);
            tensors[p + "ffn_norm.weight"] = Tensor(p + "ffn_norm.weight", [embedding]);
            tensors[p + "ffn_gate.weight"] = Tensor(p + "ffn_gate.weight", [embedding, ffn]);
            tensors[p + "ffn_up.weight"] = Tensor(p + "ffn_up.weight", [embedding, ffn]);
            tensors[p + "ffn_down.weight"] = Tensor(p + "ffn_down.weight", [ffn, embedding]);
        }
        return (new ModelConfig("qwen2", values), tensors);
    }

    private static ModelTensor Tensor(string name, int[] dims) =>
        new(name, 8, "Q8_0", true, dims, 0, 0, true);
}

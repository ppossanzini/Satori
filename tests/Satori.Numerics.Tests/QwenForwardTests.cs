using System.Text.Json;
using Satori.Core.Model;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS08: the generic CPU forward is compared against an independent float64 reference on two topologies —
/// qwen2 (grouped-query attention with QKV bias) and qwen3 (Q/K per-head norm with an explicit head
/// dimension). Requires the Q8_0 corpus: <c>python tools/download_qwen_models.py</c>.
/// </summary>
[Trait("Category", "ModelIntegration")]
public sealed class QwenForwardTests
{
    public static TheoryData<string> Models => new() { "qwen2.5-coder-1.5b-q8_0", "qwen3-4b-q8_0" };

    private static string Corpus(string name) => Path.Combine(GgufTests.Root, ".temp/qwen-corpus", name + ".gguf");
    private static string Reference(string name) => Path.Combine(GgufTests.Root, "fixtures/qwen", name + "-forward-v1.json");

    [Theory]
    [MemberData(nameof(Models))]
    public void ForwardMatchesIndependentFloat64Reference(string name)
    {
        string modelPath = Corpus(name);
        Assert.True(File.Exists(modelPath), $"Missing corpus {modelPath}: run python tools/download_qwen_models.py");
        var reference = JsonDocument.Parse(File.ReadAllText(Reference(name))).RootElement;

        var definition = GgufModelLoader.Load(modelPath);
        Assert.Empty(definition.Capabilities.Gaps);
        Assert.NotNull(definition.Shape);

        // The derived shape must agree with the reference configuration, including the explicit head dim.
        var config = reference.GetProperty("config");
        Assert.Equal(config.GetProperty("layers").GetInt32(), definition.Shape!.Layers);
        Assert.Equal(config.GetProperty("embedding").GetInt32(), definition.Shape.EmbeddingLength);
        Assert.Equal(config.GetProperty("head_count").GetInt32(), definition.Shape.HeadCount);
        Assert.Equal(config.GetProperty("head_count_kv").GetInt32(), definition.Shape.HeadCountKv);
        Assert.Equal(config.GetProperty("head_dim").GetInt32(), definition.Shape.HeadDim);
        Assert.Equal(config.GetProperty("vocabulary").GetInt32(), definition.Shape.VocabularySize);

        int[] tokens = reference.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();
        int lastPosition = tokens.Length - 1;

        var stages = new Dictionary<(int Layer, string Stage), float[]>();
        using var file = new GgufFile(modelPath);
        using var weights = new GgufWeightStore(file);
        var forward = new DenseForward(definition.Shape, weights);
        float[] logits = forward.Logits(tokens, (layer, position, stage, values) =>
        {
            if (layer >= 0 && position != lastPosition) return;
            stages[(layer, stage)] = values.ToArray();
        });

        // Argmax and the top set must match: a token-level divergence is a real defect, not tolerance.
        int argmax = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        Assert.Equal(reference.GetProperty("argmax").GetInt32(), argmax);

        // Sampled logits across the vocabulary, within a tolerance motivated by the float64 reference.
        const double absolute = 5e-3, relative = 5e-3;
        foreach (var sample in reference.GetProperty("samples").EnumerateArray())
        {
            int index = sample[0].GetInt32();
            double expected = sample[1].GetDouble();
            double error = Math.Abs(logits[index] - expected);
            Assert.True(error <= absolute + relative * Math.Abs(expected),
                $"{name} logit[{index}]: {logits[index]} vs {expected} (error {error})");
        }

        // Intermediate tensors on the first and last layer.
        foreach (var stage in reference.GetProperty("stages").EnumerateArray())
        {
            int layer = stage.GetProperty("layer").GetInt32();
            string stageName = stage.GetProperty("stage").GetString()!;
            var expected = stage.GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            Assert.True(stages.TryGetValue((layer, stageName), out var actual), $"missing stage {layer}/{stageName}");
            Assert.Equal(expected.Length, actual!.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                double error = Math.Abs(actual[i] - expected[i]);
                Assert.True(error <= 2e-3 + 2e-3 * Math.Abs(expected[i]),
                    $"{name} stage {layer}/{stageName}[{i}]: {actual[i]} vs {expected[i]}");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void IncrementalSessionMatchesFullForward(string name)
    {
        string modelPath = Corpus(name);
        Assert.True(File.Exists(modelPath), $"Missing corpus {modelPath}: run python tools/download_qwen_models.py");
        var reference = JsonDocument.Parse(File.ReadAllText(Reference(name))).RootElement;
        int[] tokens = reference.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();

        var definition = GgufModelLoader.Load(modelPath);
        Assert.Empty(definition.Capabilities.Gaps);
        using var file = new GgufFile(modelPath);
        using var weights = new GgufWeightStore(file);

        // The session reaches the same independent-reference argmax as the whole-sequence pass.
        using var session = new DenseSession(definition, weights, contextLength: 64);
        float[] logits = new float[definition.Shape!.VocabularySize];
        session.Forward(tokens, logits);
        int argmax = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        Assert.Equal(reference.GetProperty("argmax").GetInt32(), argmax);

        // Appending one token equals recomputing the whole prefix through the whole-sequence forward.
        float[] extended = new DenseForward(definition.Shape, weights).Logits([.. tokens, 100]);
        session.Append(100);
        for (int i = 0; i < logits.Length; i++)
            Assert.True(Math.Abs(extended[i] - session.Logits[i]) <= 1e-3 + 1e-3 * Math.Abs(extended[i]),
                $"{name} incremental logit[{i}]: {session.Logits[i]} vs {extended[i]}");
        Assert.Equal(tokens.Length + 1, session.Position);
    }
}

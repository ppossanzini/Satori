using System.Buffers.Binary;
using System.Text.Json;
using Satori.Core;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS07: Qwen2/Qwen3 operators checked against an independent float64 reference — grouped-query attention,
/// half-split RoPE with configurable base and rotary dimension, per-head RMS norm and bias add.
/// </summary>
public sealed class QwenOperatorTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/transformer-v2.json"))).RootElement;

    private static float[] Floats(JsonElement element) =>
        element.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    private static void Check(float[] actual, JsonElement expectedElement, string name)
    {
        var expected = expectedElement.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-6 + 2e-6 * Math.Abs(expected[i]),
                $"{name}[{i}]: {actual[i]} vs {expected[i]}");
    }

    [Fact]
    public void GroupedQueryAttentionMatchesReference()
    {
        var g = Fixture.GetProperty("gqa");
        int heads = g.GetProperty("query_heads").GetInt32();
        int kvHeads = g.GetProperty("kv_heads").GetInt32();
        int headDim = g.GetProperty("head_dim").GetInt32();
        int tokens = g.GetProperty("tokens").GetInt32();

        float[] query = Floats(g.GetProperty("query"));
        float[] keys = Floats(g.GetProperty("keys"));
        float[] values = Floats(g.GetProperty("values"));
        float[] output = new float[query.Length];
        TransformerOps.GroupedAttention(query, keys, values, heads, kvHeads, headDim, tokens, new double[tokens], output);
        Check(output, g.GetProperty("output"), "gqa");
    }

    [Fact]
    public void GroupedQueryAttentionReducesToPlainAttentionWhenHeadsEqual()
    {
        // Two query heads, two KV heads, head dim 4, three tokens.
        var rng = new Random(11);
        float[] query = Enumerable.Range(0, 8).Select(_ => (float)rng.NextDouble()).ToArray();
        float[] keys = Enumerable.Range(0, 3 * 8).Select(_ => (float)rng.NextDouble()).ToArray();
        float[] values = Enumerable.Range(0, 3 * 8).Select(_ => (float)rng.NextDouble()).ToArray();

        float[] grouped = new float[8];
        TransformerOps.GroupedAttention(query, keys, values, 2, 2, 4, 3, new double[3], grouped);
        float[] plain = new float[8];
        TransformerOps.Attention(query, keys, values, 4, 3, new double[3], plain);

        Assert.Equal(plain, grouped);
    }

    [Theory]
    [InlineData("rope_full")]
    [InlineData("rope_partial")]
    public void HalfSplitRopeMatchesReference(string key)
    {
        var r = Fixture.GetProperty(key);
        int headDim = r.GetProperty("head_dim").GetInt32();
        int position = r.GetProperty("position").GetInt32();
        double baseValue = r.GetProperty("base").GetDouble();
        int rotary = r.GetProperty("rotary").GetInt32();

        float[] vector = Floats(r.GetProperty("input"));
        TransformerOps.RopeHalf(vector, headDim, position, baseValue, rotary);
        Check(vector, r.GetProperty("output"), key);
    }

    [Fact]
    public void PerHeadRmsNormMatchesReference()
    {
        var n = Fixture.GetProperty("qk_norm");
        float[] vector = Floats(n.GetProperty("input"));
        float[] weights = Floats(n.GetProperty("weights"));
        float epsilon = n.GetProperty("epsilon").GetSingle();
        TransformerOps.RmsNormHeads(vector, weights, epsilon, weights.Length);
        Check(vector, n.GetProperty("output"), "qk_norm");
    }

    [Fact]
    public void BiasAddMatchesReference()
    {
        var b = Fixture.GetProperty("bias");
        float[] baseVec = Floats(b.GetProperty("base"));
        float[] bias = Floats(b.GetProperty("bias"));
        float[] output = new float[baseVec.Length];
        TransformerOps.Add(baseVec, bias, output);
        Check(output, b.GetProperty("output"), "bias");
    }

    [Fact]
    public void BiasedProjectionWithIndependentWidthsMatchesFp64()
    {
        // Q/K/V projections change width: the output width (rows) differs from the embedding (columns).
        // The projection reuses the conformant Q8_0 MatVec; the bias reuses TransformerOps.Add.
        const int rows = 8, columns = 32;
        var shape = new QuantizedMatrixShape(rows, columns, BlockQuantization.Q8_0);
        float[] scales = [0.5f, 1f, -2f, 0f, 0.25f, 8f, -0.5f, 3f];
        byte[] payload = new byte[shape.ByteLength];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * 34;
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)scales[row]));
            for (int i = 0; i < 32; i++) payload[offset + 2 + i] = unchecked((byte)(sbyte)(i - 16));
        }
        float[] bias = Enumerable.Range(0, rows).Select(r => (r - 3.5f) / 2f).ToArray();
        float[] input = Enumerable.Repeat(1f, columns).ToArray();

        var matrix = new QuantizedMatrix(payload, shape);
        float[] projected = new float[rows];
        matrix.Execute(input, projected);
        float[] output = new float[rows];
        TransformerOps.Add(projected, bias, output);

        Assert.Equal(rows, output.Length);
        for (int row = 0; row < rows; row++)
        {
            // Independent float64 reference from a manual decode of the same payload.
            double scale = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(row * 34)));
            double sum = 0;
            for (int i = 0; i < 32; i++) sum += (sbyte)payload[row * 34 + 2 + i];
            double expected = scale * sum + bias[row];
            Assert.True(Math.Abs(output[row] - expected) <= 1e-5 + 1e-5 * Math.Abs(expected),
                $"row {row}: {output[row]} vs {expected}");
        }
    }

    [Fact]
    public void InvalidGroupedAttentionDimensionsAreRejected()
    {
        float[] q = new float[32], k = new float[3 * 16], v = new float[3 * 16];
        Assert.Throws<ArgumentException>(() => TransformerOps.GroupedAttention(q, k, v, 4, 3, 8, 3, new double[3], new float[32]));
        Assert.Throws<ArgumentException>(() => TransformerOps.GroupedAttention(q, k, v, 4, 2, 8, 3, new double[1], new float[32]));
        Assert.Throws<ArgumentOutOfRangeException>(() => TransformerOps.RopeHalf(new float[8], 8, 0, 10000, 5));
        Assert.Throws<ArgumentException>(() => TransformerOps.RmsNormHeads(new float[8], new float[4], 1e-6f, 8));
    }
}

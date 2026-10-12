using System.Text.Json;
using Satori.Core;

namespace Satori.Numerics.Tests;

public sealed class TransformerOpsTests
{
    [Fact]
    public void OperatorsMatchIndependentFloat64Fixtures()
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/transformer-v1.json")));
        float[] Input(string key) => fixture.RootElement.GetProperty(key).EnumerateArray().Select(v => v.GetSingle()).ToArray();
        void Check(string key, float[] actual)
        {
            var expected = fixture.RootElement.GetProperty(key).EnumerateArray().Select(v => v.GetDouble()).ToArray();
            for (int i = 0; i < actual.Length; i++) Assert.True(float.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= 2e-6 + 2e-6 * Math.Abs(expected[i]), $"{key}[{i}]");
        }
        float[] x = Input("x"), output = new float[x.Length];
        TransformerOps.RmsNorm(x, Input("weights"), 1e-5f, output); Check("rms", output);
        TransformerOps.Rope(x, 48, 37); Check("rope", x);
        output = new float[8];
        TransformerOps.SwiGlu(Input("gate"), Input("up"), output); Check("swiglu", output);
        output = new float[96];
        TransformerOps.Attention(Input("query"), Input("keys"), Input("values"), 48, 7, new double[7], output); Check("attention", output);
    }

    [Fact]
    public void ExtremeFiniteValuesRemainStableAndInPlaceIsSafe()
    {
        float[] x = [float.MaxValue, -float.MaxValue], weights = [1, 1];
        TransformerOps.RmsNorm(x, weights, 1e-5f, x);
        Assert.Equal(new float[] { 1, -1 }, x);
        float[] q = [float.MaxValue, float.MaxValue], k = [float.MaxValue, float.MaxValue, -float.MaxValue, -float.MaxValue];
        float[] result = new float[2];
        TransformerOps.Attention(q, k, [3, 4, 100, 200], 2, 2, new double[2], result);
        Assert.Equal(new float[] {3, 4}, result);
        TransformerOps.SwiGlu(x, weights, x);
        Assert.All(x, value => Assert.True(float.IsFinite(value)));
        TransformerOps.Add(x, weights, x);
        Assert.All(x, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void InvalidDimensionsAndAliasingAreRejectedBeforeWriting()
    {
        float[] x = new float[8];
        Assert.Throws<ArgumentException>(() => TransformerOps.Add(x.AsSpan(0,4), x.AsSpan(0,4), x.AsSpan(1,4)));
        Assert.Throws<ArgumentException>(() => TransformerOps.Attention(x, x, x, 2, 1, new double[1], x));
        Assert.Throws<ArgumentException>(() => TransformerOps.Attention(x, x, x, 2, 2, new double[2], new float[8]));
        Assert.Throws<ArgumentOutOfRangeException>(() => TransformerOps.Rope(x, 3, 0));
    }

    [Fact]
    public void Q8RowDecodesSignedCodesAndHalfScale()
    {
        byte[] row = new byte[34]; row[0] = 0; row[1] = 0xB8; // -0.5 in binary16
        for (int i = 0; i < 32; i++) row[2+i] = unchecked((byte)(i*8-128));
        float[] output = new float[32];
        QuantizedRow.DecodeQ8(row, output);
        for (int i = 0; i < 32; i++) Assert.Equal((i*8-128)*-0.5f, output[i]);
    }
}

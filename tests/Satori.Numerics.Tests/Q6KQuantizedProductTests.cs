using System.Text.Json;
using Satori.Core;

namespace Satori.Numerics.Tests;

/// <summary>
/// The Q6_K integer product, which is the decode path's largest single cost: <c>ffn_down</c> and the output
/// projection are Q6_K and together spend more than half a decode token. The Q6_K tensors had no integer
/// kernel, so they ran the floating-point decode path at 2.5x to 3.6x less throughput per MAC.
/// </summary>
/// <remarks>
/// Two oracles, because one of them was not enough the last time this product was attempted. The fixture
/// case is an independent FP64 reproduction whose Q6_K decode and Q8_K quantizer are both checked against
/// the pinned ggml source. The in-process case composes Satori's own decoder, which an independent fixture
/// already validates, with Satori's own quantizer, which is checked byte for byte; that composition is
/// therefore anchored outside the product under test even though it runs in this process. A wrong grouping
/// of the sub-scales passes neither.
/// </remarks>
public sealed class Q6KQuantizedProductTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/q6k-q8k-v1.json"))).RootElement;

    private static byte[] Hex(string value) => Convert.FromHexString(value);

    [Fact]
    public void IntegerProductMatchesTheIndependentReference()
    {
        int cases = 0;
        foreach (var test in Fixture.GetProperty("cases").EnumerateArray())
        {
            string name = test.GetProperty("name").GetString()!;
            byte[] weights = Hex(test.GetProperty("weights_hex").GetString()!);
            byte[] activation = Hex(test.GetProperty("activation_hex").GetString()!);
            double reference = test.GetProperty("dot").GetDouble();
            float actual = Q6KCodec.DotBlockQuantized(weights, activation);
            Assert.True(Math.Abs(actual - reference) <= 1e-3 + 1e-4 * Math.Abs(reference),
                $"{name} dot: {actual} vs {reference}");
            cases++;
        }
        Assert.Equal(5, cases);
    }

    /// <summary>
    /// The product must equal the composition of the two already-validated components, computed here through
    /// the decoder rather than through the product. This is what catches a sub-scale grouping that is
    /// self-consistent but not the layout's.
    /// </summary>
    [Fact]
    public void IntegerProductEqualsTheValidatedDecoderComposition()
    {
        int cases = 0;
        foreach (var test in Fixture.GetProperty("cases").EnumerateArray())
        {
            byte[] weights = Hex(test.GetProperty("weights_hex").GetString()!);
            byte[] activation = Hex(test.GetProperty("activation_hex").GetString()!);

            float[] decoded = new float[Q6KCodec.ElementsPerBlock];
            Q6KCodec.DecodeBlock(weights, decoded);
            var codes = Q8KCodec.Codes(activation);
            float scale = Q8KCodec.Scale(activation);
            double expected = 0;
            for (int i = 0; i < Q6KCodec.ElementsPerBlock; i++) expected += (double)decoded[i] * codes[i];
            expected *= scale;

            float actual = Q6KCodec.DotBlockQuantized(weights, activation);
            Assert.True(Math.Abs(actual - expected) <= 1e-3 + 1e-4 * Math.Abs(expected),
                $"composition: {actual} vs {expected}");
            cases++;
        }
        Assert.Equal(5, cases);
    }

    /// <summary>
    /// The centring of the six-bit codes is the trap: the reference kernel multiplies uncentred codes and
    /// removes the offset through group sums, while the decoder subtracts 32 per weight. A product that got
    /// this wrong would be off by <c>32 * d * Σ scale * code</c>, so the check is that the result differs
    /// from the uncentred form by exactly that amount.
    /// </summary>
    [Fact]
    public void TheCodeOffsetIsRemovedAndNotJustDropped()
    {
        int cases = 0;
        foreach (var test in Fixture.GetProperty("cases").EnumerateArray())
        {
            byte[] weights = Hex(test.GetProperty("weights_hex").GetString()!);
            byte[] activation = Hex(test.GetProperty("activation_hex").GetString()!);
            var codes = Q8KCodec.Codes(activation);
            float scale = Q8KCodec.Scale(activation);
            float d = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(weights, 208));

            float[] decoded = new float[Q6KCodec.ElementsPerBlock];
            Q6KCodec.DecodeBlock(weights, decoded);
            // Σ decoded * code is d * Σ scale*(q-32) * code. The uncentred form would have used q, i.e.
            // scale*(q-32) + 32*scale, so the difference is 32*d*scale*Σ scale*code over each sub-block.
            double offset = 0;
            var scales = weights.AsSpan(192, 16);
            for (int sub = 0; sub < 16; sub++)
            {
                int sum = 0;
                for (int i = 0; i < 16; i++) sum += codes[sub * 16 + i];
                offset += 32.0 * scale * d * (sbyte)scales[sub] * sum;
            }
            double centered = 0;
            for (int i = 0; i < Q6KCodec.ElementsPerBlock; i++) centered += (double)decoded[i] * codes[i];
            centered *= scale;

            float actual = Q6KCodec.DotBlockQuantized(weights, activation);
            Assert.True(Math.Abs(actual - (centered)) <= 1e-3 + 1e-4 * Math.Abs(centered),
                $"centred: {actual} vs {centered}");
            // The offset must be non-trivial on at least one case, or this check proves nothing.
            if (Math.Abs(offset) > 1e-3) cases++;
        }
        Assert.True(cases > 0, "no fixture case exercises the -32 centring.");
    }

    /// <summary>
    /// A block the quantizer found nothing to represent gets a zero scale, and the scaled product must be
    /// zero. The raw integer product is deliberately **not** asserted to be zero: it is the sum of the codes
    /// against the weights and carries no meaning until the scale is applied, which is exactly why the kernel
    /// skips such a block instead of computing it. Asserting it here would be asserting a coincidence.
    /// </summary>
    [Fact]
    public void ZeroActivationBlockGivesAZeroScaledProduct()
    {
        byte[] weights = Hex(Fixture.GetProperty("cases")[0].GetProperty("weights_hex").GetString()!);
        byte[] activation = new byte[Q8KCodec.BytesPerBlock];
        activation.AsSpan(Q8KCodec.QuantsOffset, Q8KCodec.ElementsPerBlock).Fill(0x7F);
        Assert.True(Q8KCodec.IsZero(activation));
        Assert.Equal(0f, Q6KCodec.DotBlockQuantized(weights, activation));

        // And an activation block the quantizer itself produced from all zeros must be the same case.
        Q8KCodec.QuantizeBlock(new float[Q8KCodec.ElementsPerBlock], activation);
        Assert.True(Q8KCodec.IsZero(activation));
        Assert.Equal(0f, Q6KCodec.DotBlockQuantized(weights, activation));
    }

    [Fact]
    public void MismatchedLengthsAreRejected()
    {
        byte[] weights = new byte[Q6KCodec.BytesPerBlock];
        byte[] activation = new byte[Q8KCodec.BytesPerBlock];
        Assert.Throws<ArgumentException>(() => Q6KCodec.DotProduct(weights[..100], activation));
        Assert.Throws<ArgumentException>(() => Q6KCodec.DotProduct(weights, activation[..100]));
        Assert.Throws<ArgumentException>(() => Q6KCodec.DotBlockQuantized(weights[..100], activation));
    }
}

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

    /// <summary>
    /// The vectorised kernel must return the scalar product's integer exactly, block for block. Both compute
    /// <c>Σ scale_s * Σ (q - 32) * code</c>, which is a sum of integer products, so agreement is required bit
    /// for bit and a difference is a defect rather than a rounding artefact. This is the check the scalar form
    /// could not make on the vector form: the first attempt at this kernel produced a sign error that a
    /// tolerance-based test would have absorbed.
    /// </summary>
    /// <remarks>
    /// The activations are real Q8_K blocks and not arbitrary bytes, because a random activation byte pattern
    /// gives a scale that is not a finite number and the kernel would then compare two infinities. The test
    /// says so rather than relying on the reader to notice.
    /// </remarks>
    [Fact]
    public void TheVectorisedProductEqualsTheScalarProductExactly()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            return;
        }
        const int blocks = 200;
        var random = new Random(20261014);
        float[] activationValues = new float[Q8KCodec.ElementsPerBlock];
        byte[] activation = new byte[Q8KCodec.BytesPerBlock];
        int compared = 0;
        for (int block = 0; block < blocks; block++)
        {
            byte[] weights = new byte[Q6KCodec.BytesPerBlock];
            random.NextBytes(weights);
            // Six-bit quants are six bits; the two spare bits of each byte belong to no quant and the decoder
            // ignores them, so leaving them random is the realistic case and also the harder one.
            for (int index = 0; index < activationValues.Length; index++)
                activationValues[index] = (float)(random.NextDouble() * 2 - 1);
            Q8KCodec.QuantizeBlock(activationValues, activation);
            if (Q8KCodec.IsZero(activation))
            {
                continue;
            }

            int scalar = Q6KCodec.DotProduct(weights, activation);
            int vector;
            unsafe
            {
                fixed (byte* weightsPointer = weights)
                fixed (byte* activationPointer = activation)
                    vector = Satori.Core.QuantizedKAvx2Kernel.DotProductQ6K(weightsPointer, activationPointer);
            }
            Assert.Equal(scalar, vector);
            compared++;
        }
        Assert.True(compared > blocks / 2, $"Only {compared} of {blocks} blocks were comparable.");
    }

    /// <summary>
    /// The same equality on the artifact's own Q6_K blocks, where the scales and quants come from a real model
    /// rather than a generator. Skipped when the artifact is absent, like the other real-bytes tests.
    /// </summary>
    [Fact]
    public void TheVectorisedProductEqualsTheScalarProductOnRealBlocks()
    {
        const string artifact = ".temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
        string path = Path.Combine(GgufTests.Root, artifact);
        if (!File.Exists(path))
        {
            return;
        }
        using var file = new Satori.Formats.Gguf.GgufFile(path);
        var random = new Random(20261014);
        float[] activationValues = new float[Q8KCodec.ElementsPerBlock];
        byte[] activation = new byte[Q8KCodec.BytesPerBlock];
        int compared = 0;
        foreach (string name in new[] { "blk.0.ffn_down.weight", "output.weight" })
        {
            var matrix = new Satori.Formats.Gguf.GgufQuantizedMatrix(file, name);
            Assert.Equal(BlockQuantization.Q6_K, matrix.Shape.Format);
            byte[] payload = new byte[matrix.Shape.ByteLength];
            file.ReadTensor(name, 0, payload.Length, static (bytes, target) => bytes.CopyTo(target), payload);
            int blocksPerRow = matrix.Shape.Columns / Q6KCodec.ElementsPerBlock;
            for (int row = 0; row < matrix.Shape.Rows; row += 97)
            {
                for (int columnBlock = 0; columnBlock < blocksPerRow; columnBlock++)
                {
                    int offset = (row * blocksPerRow + columnBlock) * Q6KCodec.BytesPerBlock;
                    for (int index = 0; index < activationValues.Length; index++)
                        activationValues[index] = (float)(random.NextDouble() * 2 - 1);
                    Q8KCodec.QuantizeBlock(activationValues, activation);
                    int scalar = Q6KCodec.DotProduct(payload.AsSpan(offset, Q6KCodec.BytesPerBlock), activation);
                    int vector;
                    unsafe
                    {
                        fixed (byte* weightsPointer = payload.AsSpan(offset, Q6KCodec.BytesPerBlock))
                        fixed (byte* activationPointer = activation)
                            vector = Satori.Core.QuantizedKAvx2Kernel.DotProductQ6K(weightsPointer, activationPointer);
                    }
                    Assert.Equal(scalar, vector);
                    compared++;
                }
            }
        }
        Assert.True(compared > 100, $"Only {compared} real blocks were compared.");
    }
}

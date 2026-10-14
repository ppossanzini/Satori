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
    /// The projection's Q6_K integer path must equal, bit for bit, the sum its own public codecs describe:
    /// the Q8_K quantizer the kernel consumes, and the Q6_K codec whose decoder an independent fixture
    /// validates. The projection is the public entry point and the reference is assembled from public pieces,
    /// so the vectorised product is held to account through the API the product offers rather than through a
    /// method that exists only to be measured.
    /// </summary>
    /// <remarks>
    /// The comparison is exact rather than tolerant. Every term is a sum of integer products, so the two must
    /// agree on every bit, and a tolerance would have absorbed the sign error that the first attempt at this
    /// kernel produced. Both sides add the same per-block products in the same order and in the same
    /// expression, so there is no rounding difference for a tolerance to excuse.
    /// </remarks>
    [Fact]
    public void TheProjectionEqualsTheCompositionOfThePublicCodecs()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            return;
        }
        const int rows = 48, columns = 1024;
        var shape = new QuantizedMatrixShape(rows, columns, BlockQuantization.Q6_K);
        byte[] payload = new byte[shape.ByteLength];
        var random = new Random(20261014);
        random.NextBytes(payload);
        int blocksPerRow = columns / Q6KCodec.ElementsPerBlock;
        for (int row = 0; row < rows; row++)
            for (int block = 0; block < blocksPerRow; block++)
                WriteScale(payload, (row * blocksPerRow + block) * Q6KCodec.BytesPerBlock);

        float[] input = new float[columns];
        for (int index = 0; index < columns; index++) input[index] = (float)Math.Sin(index * 0.017) * 0.5f;
        float[] actual = new float[rows];
        QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, actual, threads: 8);

        for (int row = 0; row < rows; row++)
            Assert.Equal(ComposeRow(payload, shape, input, row), actual[row]);
    }

    /// <summary>
    /// The same equality on the artifact's own Q6_K tensors, where the quants and sub-scales come from a real
    /// model rather than a generator. Skipped when the artifact is absent, like the other real-bytes tests.
    /// </summary>
    [Fact]
    public void TheProjectionEqualsTheCompositionOfThePublicCodecsOnRealTensors()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            return;
        }
        const string artifact = ".temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
        string path = Path.Combine(GgufTests.Root, artifact);
        if (!File.Exists(path))
        {
            return;
        }
        using var file = new Satori.Formats.Gguf.GgufFile(path);
        foreach (string name in new[] { "blk.0.ffn_down.weight", "output.weight" })
        {
            var matrix = new Satori.Formats.Gguf.GgufQuantizedMatrix(file, name);
            Assert.Equal(BlockQuantization.Q6_K, matrix.Shape.Format);
            byte[] payload = new byte[matrix.Shape.ByteLength];
            file.ReadTensor(name, 0, payload.Length, static (bytes, target) => bytes.CopyTo(target), payload);
            int columns = matrix.Shape.Columns;
            float[] input = new float[columns];
            for (int index = 0; index < columns; index++) input[index] = (float)Math.Sin(index * 0.013) * 0.5f;
            float[] actual = new float[matrix.Shape.Rows];
            QuantizedMatVec.ExecuteBatch(payload, matrix.Shape, input, 1, actual, threads: 8);

            // Every forty-first row: enough of a real tensor to exercise the mapping, without timing the whole
            // 152064-row projection four times over.
            for (int row = 0; row < matrix.Shape.Rows; row += 41)
                Assert.Equal(ComposeRow(payload, matrix.Shape, input, row), actual[row]);
        }
    }

    /// <summary>
    /// One row's output, summed block by block from the two public codecs exactly as the projection sums it.
    /// This is the reference the projection is held to, written without reaching inside it.
    /// </summary>
    private static float ComposeRow(byte[] payload, QuantizedMatrixShape shape, float[] input, int row)
    {
        int blocksPerRow = shape.Columns / Q6KCodec.ElementsPerBlock;
        float total = 0;
        for (int block = 0; block < blocksPerRow; block++)
        {
            var activationBlock = new byte[Q8KCodec.BytesPerBlock];
            Q8KCodec.QuantizeBlock(input.AsSpan(block * Q6KCodec.ElementsPerBlock, Q6KCodec.ElementsPerBlock),
                activationBlock);
            total += Q6KCodec.DotBlockQuantized(
                payload.AsSpan((row * blocksPerRow + block) * Q6KCodec.BytesPerBlock, Q6KCodec.BytesPerBlock),
                activationBlock);
        }
        return total;
    }

    /// <summary>Writes a finite block scale, without which the product is a multiplication by whatever bytes
    /// happen to sit there.</summary>
    private static void WriteScale(byte[] payload, int blockOffset) =>
        BitConverter.TryWriteBytes(
            payload.AsSpan(blockOffset + Q6KCodec.BytesPerBlock - 2, sizeof(ushort)), (Half)0.01f);
}

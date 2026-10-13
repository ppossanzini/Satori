using System.Text.Json;
using Satori.Core;

namespace Satori.Numerics.Tests;

/// <summary>
/// DW06.4: parity of the K codecs on real artifact bytes. The fixture is produced by
/// <c>tools/k_tensor_reference.py</c> from the pinned Qwen2.5-Coder 7B Q4_K_M tensors
/// (<c>token_embd.weight</c> Q4_K, <c>output.weight</c> Q6_K) and an independent FP64 decode, so a matching
/// result means the Satori decoder reproduces the reference on the same bytes the runner uses.
/// </summary>
public sealed class KParityTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/k-real-v1.json"))).RootElement;

    private static byte[] Hex(string value) => Convert.FromHexString(value);

    [Fact]
    public void RealTensorBlocksDecodeToTheIndependentReference()
    {
        foreach (var test in Fixture.GetProperty("cases").EnumerateArray())
        {
            string name = test.GetProperty("name").GetString()!;
            bool isQ4K = name.StartsWith("q4k:", StringComparison.Ordinal);
            int blockBytes = isQ4K ? Q4KCodec.BytesPerBlock : Q6KCodec.BytesPerBlock;
            int blocks = test.GetProperty("blocks").GetInt32();
            byte[] raw = Hex(test.GetProperty("block_hex").GetString()!);
            Assert.Equal(blocks * blockBytes, raw.Length);

            var referenceWeights = test.GetProperty("weights").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            float[] weights = new float[blocks * 256];
            for (int b = 0; b < blocks; b++)
            {
                var slice = weights.AsSpan(b * 256, 256);
                if (isQ4K) Q4KCodec.DecodeBlock(raw.AsSpan(b * blockBytes, blockBytes), slice);
                else Q6KCodec.DecodeBlock(raw.AsSpan(b * blockBytes, blockBytes), slice);
            }
            for (int i = 0; i < weights.Length; i++)
                Assert.True(Math.Abs(weights[i] - referenceWeights[i]) <= 1e-3 + 1e-4 * Math.Abs(referenceWeights[i]),
                    $"{name} weight[{i}]: {weights[i]} vs {referenceWeights[i]}");

            float[] input = test.GetProperty("input").EnumerateArray().Select(v => v.GetSingle()).ToArray();
            var dots = test.GetProperty("dots").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            for (int b = 0; b < blocks; b++)
            {
                float dot = isQ4K
                    ? Q4KCodec.DotBlock(raw.AsSpan(b * blockBytes, blockBytes), input)
                    : Q6KCodec.DotBlock(raw.AsSpan(b * blockBytes, blockBytes), input);
                Assert.True(Math.Abs(dot - dots[b]) <= 1e-2 + 1e-4 * Math.Abs(dots[b]),
                    $"{name} dot[{b}]: {dot} vs {dots[b]}");
            }
        }
    }

    [Theory]
    [InlineData("fixtures/operators/q4k-v1.json", BlockQuantization.Q4_K)]
    [InlineData("fixtures/operators/q6k-v1.json", BlockQuantization.Q6_K)]
    public void Avx2MatchesScalarOnEveryBlock(string fixture, BlockQuantization format)
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(GgufTests.Root, fixture)))
            .RootElement.GetProperty("cases").EnumerateArray().ToArray();

        // Two blocks per row exercises a multi-super-block row; the same input vector is reused.
        foreach (var (first, second) in new[] { (cases[0], cases[1]), (cases[4], cases[7]) })
        {
            byte[] payload = [.. Hex(first.GetProperty("block_hex").GetString()!), .. Hex(second.GetProperty("block_hex").GetString()!)];
            float[] input = Enumerable.Range(0, 512).Select(i => (float)((i % 17) - 8) * 0.5f).ToArray();
            var shape = new QuantizedMatrixShape(1, 512, format);
            Assert.Equal(payload.Length, shape.ByteLength);

            float[] scalar = new float[1];
            float[] vector = new float[1];
            QuantizedMatVec.ExecuteScalar(payload, shape, input, scalar);
            QuantizedMatVec.ExecuteAvx2(payload, shape, input, vector);
            Assert.True(Math.Abs(scalar[0] - vector[0]) <= 1e-3 + 1e-4 * Math.Abs(scalar[0]),
                $"{format}: scalar {scalar[0]} vs AVX2 {vector[0]}");
        }
    }
}

using System.Text.Json;
using Satori.Core;

namespace Satori.Numerics.Tests;

/// <summary>
/// DW06.6: the batched projection must agree with the single-position projection on the same bytes, and the
/// row split must not change the result. Fixtures are the DW06.1 and DW06.2 Q4_K/Q6_K blocks, so the codec
/// under test is the real one; the multi-row payloads are built by repeating those blocks, and the
/// independent reference is the per-position path itself.
/// </summary>
public sealed class BatchedMatVecTests
{
    private static readonly JsonElement Q4K = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/q4k-v1.json"))).RootElement;

    private static readonly JsonElement Q6K = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(GgufTests.Root, "fixtures/operators/q6k-v1.json"))).RootElement;

    /// <summary>Bytes of every fixture case of a file, which is one valid block each.</summary>
    private static List<byte[]> Blocks(JsonElement fixture) =>
        [.. fixture.GetProperty("cases").EnumerateArray().Select(c => Convert.FromHexString(c.GetProperty("block_hex").GetString()!))];

    /// <summary>A full payload built by repeating the fixture blocks, so every block is a known-good one.</summary>
    private static byte[] Payload(IReadOnlyList<byte[]> blocks, int rows, int columns, int blockBytes)
    {
        int blockElements = blockBytes == Q4KCodec.BytesPerBlock ? Q4KCodec.ElementsPerBlock : Q6KCodec.ElementsPerBlock;
        int blocksPerRow = columns / blockElements;
        byte[] payload = new byte[checked(rows * blocksPerRow * blockBytes)];
        for (int index = 0; index < rows * blocksPerRow; index++)
            blocks[index % blocks.Count].CopyTo(payload, index * blockBytes);
        return payload;
    }

    /// <summary>Deterministic input laid out one vector per position, as <c>ExecuteBatch</c> expects.</summary>
    private static float[] Input(int positions, int columns)
    {
        float[] values = new float[checked(positions * columns)];
        for (int position = 0; position < positions; position++)
            for (int column = 0; column < columns; column++)
                values[position * columns + column] = (float)(Math.Sin(position * 0.37 + column * 0.011) * 1.5);
        return values;
    }

    public static TheoryData<int, int, int, BlockQuantization> Cases => new()
    {
        { 2, 512, 4, BlockQuantization.Q4_K },
        { 3, 512, 1, BlockQuantization.Q4_K },
        { 2, 256, 8, BlockQuantization.Q6_K },
    };

    /// <summary>
    /// The batched Q4_K path quantizes the activation to 8 bits before multiplying, so it is not expected to
    /// reproduce the floating-point path bit for bit: an 8-bit code carries about 1/254 of the activation's
    /// magnitude, and over a 256-term sum of mixed signs the relative error on the result lands around 1e-4.
    /// The tolerance below is that bound with headroom, not a tuned number; the Q4_K codec itself has its own
    /// exact tests against the independent reference. Q6_K is not on this path and stays on the floating path.
    /// </summary>
    private const double QuantizedTolerance = 1e-2;

    private const double QuantizedRelativeTolerance = 1e-3;

    [Theory]
    [MemberData(nameof(Cases))]
    public void BatchMatchesPerPositionProjection(int rows, int columns, int positions, BlockQuantization format)
    {
        bool k = format == BlockQuantization.Q4_K;
        byte[] payload = Payload(k ? Blocks(Q4K) : Blocks(Q6K), rows, columns, k ? Q4KCodec.BytesPerBlock : Q6KCodec.BytesPerBlock);
        var shape = new QuantizedMatrixShape(rows, columns, format);
        Assert.Equal(payload.Length, shape.ByteLength);
        float[] input = Input(positions, columns);

        float[] batched = new float[checked(positions * rows)];
        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 4);

        float[] single = new float[rows];
        for (int position = 0; position < positions; position++)
        {
            QuantizedMatVec.ExecuteScalar(payload, shape, input.AsSpan(position * columns, columns), single);
            for (int row = 0; row < rows; row++)
            {
                float expected = single[row];
                float actual = batched[position * rows + row];
                Assert.True(Math.Abs(expected - actual) <= QuantizedTolerance + QuantizedRelativeTolerance * Math.Abs(expected),
                    $"position {position} row {row}: {actual} vs {expected}");
            }
        }
    }

    /// <summary>The row split is only an assignment of work, so every thread count must produce the same bytes.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(32)]
    public void RowSplitDoesNotChangeTheResult(int threads)
    {
        byte[] payload = Payload(Blocks(Q4K), rows: 7, columns: 512, Q4KCodec.BytesPerBlock);
        var shape = new QuantizedMatrixShape(7, 512, BlockQuantization.Q4_K);
        float[] input = Input(5, 512);
        float[] first = new float[35];
        QuantizedMatVec.ExecuteBatch(payload, shape, input, 5, first, threads: 1);
        float[] again = new float[35];
        QuantizedMatVec.ExecuteBatch(payload, shape, input, 5, again, threads);
        Assert.Equal(first, again);
    }

    [Fact]
    public void MismatchedDimensionsAreRejected()
    {
        byte[] payload = Payload(Blocks(Q4K), rows: 2, columns: 512, Q4KCodec.BytesPerBlock);
        var shape = new QuantizedMatrixShape(2, 512, BlockQuantization.Q4_K);
        float[] input = Input(3, 512);
        float[] output = new float[6];
        Assert.Throws<ArgumentException>(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 2, output, 4));
        Assert.Throws<ArgumentException>(() => QuantizedMatVec.ExecuteBatch(payload, shape, input[..512], 3, output, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 0, output, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 3, output, 0));
    }

    [Fact]
    public void RowDecodeOfTheSameBlockAgreesWithTheBlockDecode()
    {
        foreach (byte[] block in Blocks(Q6K))
        {
            float[] blockDecoded = new float[Q6KCodec.ElementsPerBlock];
            QuantizedRow.DecodeBlock(block, BlockQuantization.Q6_K, blockDecoded);
            float[] rowDecoded = new float[Q6KCodec.ElementsPerBlock];
            QuantizedRow.DecodeRow(block, new QuantizedMatrixShape(1, Q6KCodec.ElementsPerBlock, BlockQuantization.Q6_K), rowDecoded);
            Assert.Equal(blockDecoded, rowDecoded);
        }
    }
}

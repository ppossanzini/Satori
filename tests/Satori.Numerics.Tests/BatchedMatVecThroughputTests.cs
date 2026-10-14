using System.Diagnostics;
using Satori.Core;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

/// <summary>
/// DW06.6 measurement guard: the batched projection exists to be faster than one projection per position on
/// the same bytes. This asserts that ratio on a synthetic Q4_K matrix shaped like a real projection, and
/// prints both rates, so a change that makes the batched path slower fails here instead of only in a
/// benchmark report. It carries no numerical claim; correctness is in <see cref="BatchedMatVecTests"/>.
/// </summary>
public sealed class BatchedMatVecThroughputTests
{
    private readonly ITestOutputHelper _output;

    public BatchedMatVecThroughputTests(ITestOutputHelper output) => _output = output;

    /// <summary>Realistic coding-model projection: 3584 columns in, 3584 rows out.</summary>
    private static (byte[] Payload, QuantizedMatrixShape Shape) Matrix(int rows, int columns)
    {
        var shape = new QuantizedMatrixShape(rows, columns, BlockQuantization.Q4_K);
        byte[] payload = new byte[shape.ByteLength];
        var random = new Random(20261014);
        random.NextBytes(payload);
        // Keep the FP16 super-scales finite so the payload is a legal one.
        int blocksPerRow = columns / Q4KCodec.ElementsPerBlock;
        for (int row = 0; row < rows; row++)
            for (int block = 0; block < blocksPerRow; block++)
            {
                int offset = (row * blocksPerRow + block) * Q4KCodec.BytesPerBlock;
                BitConverter.TryWriteBytes(payload.AsSpan(offset, 2), (Half)0.01f);
                BitConverter.TryWriteBytes(payload.AsSpan(offset + 2, 2), (Half)0.005f);
            }
        return (payload, shape);
    }

    private static float[] Input(int positions, int columns)
    {
        float[] values = new float[checked(positions * columns)];
        for (int index = 0; index < values.Length; index++) values[index] = (float)Math.Sin(index * 0.017) * 0.5f;
        return values;
    }

    /// <summary>
    /// The shapes one decode token actually projects, at one position. This is the breakdown that decides
    /// where the remaining decode time goes: if a real shape is far off the rate of the square projection, the
    /// per-call cost dominates and the call count is the problem.
    /// </summary>
    [Theory]
    [InlineData("q/k/v/o", 3584, 3584)]
    [InlineData("k/v", 512, 3584)]
    [InlineData("gate/up", 18944, 3584)]
    [InlineData("down", 3584, 18944)]
    [InlineData("output", 152064, 3584)]
    public void RealDecodeShapesAtOnePosition(string name, int rows, int columns)
    {
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(1, columns);
        float[] output = new float[rows];

        QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8);
        var watch = Stopwatch.StartNew();
        QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8);
        double batchedMs = watch.Elapsed.TotalMilliseconds;

        QuantizedMatVec.Execute(payload, shape, input, output);
        watch.Restart();
        QuantizedMatVec.Execute(payload, shape, input, output);
        double perPositionMs = watch.Elapsed.TotalMilliseconds;

        double gmac = (double)rows * columns / 1e9;
        _output.WriteLine($"{name,-10} {rows,6}x{columns}: batched {batchedMs,7:F3} ms ({gmac * 1000 / batchedMs,5:F1} GMAC/s), " +
            $"per-position {perPositionMs,7:F3} ms ({gmac * 1000 / perPositionMs,5:F1} GMAC/s), ratio {perPositionMs / batchedMs:F2}x");
        Assert.True(batchedMs > 0 && perPositionMs > 0);
    }

    /// <summary>
    /// One full decode token's projections, measured as a whole. There are 197 of them per token, so the sum
    /// of the parts is the number that has to beat the decode threshold, not any single projection.
    /// </summary>
    [Fact]
    public void OneDecodeTokenOfProjections()
    {
        (int Rows, int Columns)[] perLayer = [(3584, 3584), (512, 3584), (512, 3584), (3584, 3584),
            (18944, 3584), (18944, 3584), (3584, 18944)];
        var matrices = perLayer.Select(s => Matrix(s.Rows, s.Columns)).ToArray();
        float[] input = Input(1, 3584);
        var watch = Stopwatch.StartNew();
        double totalMacs = 0;
        for (int layer = 0; layer < 28; layer++)
        {
            for (int index = 0; index < perLayer.Length; index++)
            {
                var (payload, shape) = matrices[index];
                var source = index >= 4 ? Input(1, shape.Columns) : input;
                float[] output = new float[shape.Rows];
                QuantizedMatVec.ExecuteBatch(payload, shape, source, 1, output, threads: 8);
                totalMacs += (double)shape.Rows * shape.Columns;
            }
        }
        double milliseconds = watch.Elapsed.TotalMilliseconds;
        var (outPayload, outShape) = Matrix(152064, 3584);
        float[] logits = new float[outShape.Rows];
        QuantizedMatVec.ExecuteBatch(outPayload, outShape, input, 1, logits, threads: 8);
        totalMacs += (double)outShape.Rows * outShape.Columns;
        milliseconds = watch.Elapsed.TotalMilliseconds;
        _output.WriteLine($"one token of projections: {milliseconds:F1} ms, {totalMacs / 1e9:F2} GMAC, " +
            $"{totalMacs / 1e9 / (milliseconds / 1000):F1} GMAC/s, {1000 / milliseconds:F2} tok/s ceiling");
        Assert.True(milliseconds > 0);
    }

    /// <summary>
    /// The decode case: one position. This is what decides whether a single-token projection should go through
    /// the batched kernel at all, because the batched kernel pays an activation quantization that a projection
    /// per position does not.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(16)]
    public void SinglePositionBatchedIsComparedWithThePerPositionKernel(int positions)
    {
        const int rows = 3584, columns = 3584;
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(positions, columns);
        float[] batched = new float[positions * rows];
        float[] single = new float[rows];

        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 8);
        var watch = Stopwatch.StartNew();
        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 8);
        double batchedMs = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        for (int position = 0; position < positions; position++)
            QuantizedMatVec.Execute(payload, shape, input.AsSpan(position * columns, columns), single);
        double perPositionMs = watch.Elapsed.TotalMilliseconds;

        double macs = (double)rows * columns * positions;
        _output.WriteLine($"positions={positions,3}: batched {batchedMs,7:F2} ms ({macs / batchedMs / 1e6,5:F1} GMAC/s), " +
            $"per-position {perPositionMs,7:F2} ms ({macs / perPositionMs / 1e6,5:F1} GMAC/s), " +
            $"ratio {perPositionMs / batchedMs:F2}x");
        Assert.True(batchedMs > 0 && perPositionMs > 0);
    }

    [Fact]
    public void BatchBeatsOneProjectionPerPosition()
    {
        const int rows = 3584, columns = 3584, positions = 64;
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(positions, columns);
        float[] batched = new float[positions * rows];
        float[] single = new float[rows];

        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 1);

        var watch = Stopwatch.StartNew();
        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 1);
        double batchedMs = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        for (int position = 0; position < positions; position++)
            QuantizedMatVec.ExecuteScalar(payload, shape, input.AsSpan(position * columns, columns), single);
        double scalarMs = watch.Elapsed.TotalMilliseconds;

        watch.Restart();
        for (int position = 0; position < positions; position++)
            QuantizedMatVec.Execute(payload, shape, input.AsSpan(position * columns, columns), single);
        double avx2Ms = watch.Elapsed.TotalMilliseconds;

        double macs = (double)rows * columns * positions / 1e9;
        _output.WriteLine($"batched   {batchedMs,8:F1} ms ({macs / (batchedMs / 1000),6:F1} GMAC/s)");
        _output.WriteLine($"scalar    {scalarMs,8:F1} ms ({macs / (scalarMs / 1000),6:F1} GMAC/s)  speedup {scalarMs / batchedMs:F2}x");
        _output.WriteLine($"avx2/pos  {avx2Ms,8:F1} ms ({macs / (avx2Ms / 1000),6:F1} GMAC/s)  speedup {avx2Ms / batchedMs:F2}x");

        Assert.True(batchedMs < scalarMs,
            $"Batched projection must beat {positions} scalar projections: {batchedMs:F1} ms vs {scalarMs:F1} ms.");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void ThreadsScaleTheBatchedProjection(int threads)
    {
        const int rows = 3584, columns = 3584, positions = 64;
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(positions, columns);
        float[] batched = new float[positions * rows];

        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads);

        var watch = Stopwatch.StartNew();
        QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads);
        double milliseconds = watch.Elapsed.TotalMilliseconds;
        double gmac = (double)rows * columns * positions / 1e9 / (milliseconds / 1000);
        _output.WriteLine($"threads={threads}: {milliseconds:F1} ms ({gmac:F1} GMAC/s)");
        Assert.True(gmac > 0);
    }
}

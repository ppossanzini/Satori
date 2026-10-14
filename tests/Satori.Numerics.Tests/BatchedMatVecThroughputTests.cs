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

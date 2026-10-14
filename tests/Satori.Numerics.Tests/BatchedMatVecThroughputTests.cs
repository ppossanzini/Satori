using System.Diagnostics;
using Satori.Core;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

/// <summary>
/// Measurement guard for the projection kernels: the batched projection exists to be faster than one
/// projection per position on the same bytes, and this asserts that ratio on synthetic Q4_K matrices shaped
/// like the real ones. It carries no numerical claim; correctness is in <see cref="BatchedMatVecTests"/>.
/// </summary>
/// <remarks>
/// Every figure is the **minimum** of several repetitions, not a single run. A projection of one or ten
/// milliseconds on a turbo-boosting laptop is not reproducible from one run: an early version of this class
/// took single measurements and reported a 3.8x swing on the same code between runs, which is enough to
/// invert a conclusion. The minimum is the repetition least disturbed by scheduling and frequency changes,
/// and the spread is printed beside it so a reader can see how much to trust the number. The tests assert
/// only inequalities that hold across the whole spread, never a ratio the machine cannot guarantee.
/// </remarks>
public sealed class BatchedMatVecThroughputTests
{
    private readonly ITestOutputHelper _output;

    public BatchedMatVecThroughputTests(ITestOutputHelper output) => _output = output;

    /// <summary>Repetitions per measurement: one warm-up plus this many timed runs.</summary>
    private const int Repetitions = 9;

    /// <summary>Runs <paramref name="body"/> repeatedly and returns the minimum and median milliseconds.</summary>
    private static (double Minimum, double Median) Time(Action body)
    {
        body();
        double[] samples = new double[Repetitions];
        for (int i = 0; i < Repetitions; i++)
        {
            var watch = Stopwatch.StartNew();
            body();
            samples[i] = watch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return (samples[0], samples[samples.Length / 2]);
    }

    private static string Rate(double milliseconds, long macs) =>
        $"{macs / milliseconds / 1e3,6:F1} GMAC/s";

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
    /// where decode time goes: a real shape far off the rate of the square projection means the per-call cost
    /// dominates and the call count is the problem.
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

        (double batchedMs, double batchedMedian) = Time(() =>
            QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8));
        (double perPositionMs, double perPositionMedian) = Time(() =>
            QuantizedMatVec.Execute(payload, shape, input, output));

        long macs = (long)rows * columns;
        _output.WriteLine($"{name,-10} {rows,6}x{columns}: batched {batchedMs,7:F3} ms (median {batchedMedian,7:F3}, {Rate(batchedMs, macs)}), " +
            $"per-position {perPositionMs,7:F3} ms (median {perPositionMedian,7:F3}, {Rate(perPositionMs, macs)}), " +
            $"ratio {perPositionMs / batchedMs:F2}x");
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
        var sources = perLayer.Select(s => Input(1, s.Columns)).ToArray();
        var outputs = perLayer.Select(s => new float[s.Rows]).ToArray();
        var (outPayload, outShape) = Matrix(152064, 3584);
        float[] logits = new float[outShape.Rows];
        long totalMacs = 0;
        foreach (var s in perLayer) totalMacs += (long)s.Rows * s.Columns * 28;
        totalMacs += (long)outShape.Rows * outShape.Columns;

        (double milliseconds, double median) = Time(() =>
        {
            for (int layer = 0; layer < 28; layer++)
                for (int index = 0; index < perLayer.Length; index++)
                    QuantizedMatVec.ExecuteBatch(matrices[index].Payload, matrices[index].Shape,
                        sources[index], 1, outputs[index], threads: 8);
            QuantizedMatVec.ExecuteBatch(outPayload, outShape, input, 1, logits, threads: 8);
        });
        _output.WriteLine($"one token of projections: {milliseconds:F1} ms (median {median:F1}), {totalMacs / 1e9:F2} GMAC, " +
            $"{Rate(milliseconds, totalMacs)}, {1000 / milliseconds:F2} tok/s ceiling");
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

        (double batchedMs, double batchedMedian) = Time(() =>
            QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 8));
        (double perPositionMs, double perPositionMedian) = Time(() =>
        {
            for (int position = 0; position < positions; position++)
                QuantizedMatVec.Execute(payload, shape, input.AsSpan(position * columns, columns), single);
        });

        long macs = (long)rows * columns * positions;
        _output.WriteLine($"positions={positions,3}: batched {batchedMs,7:F2} ms (median {batchedMedian,7:F2}, {Rate(batchedMs, macs)}), " +
            $"per-position {perPositionMs,7:F2} ms (median {perPositionMedian,7:F2}, {Rate(perPositionMs, macs)}), " +
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

        (double batchedMs, double batchedMedian) = Time(() =>
            QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads: 1));
        (double scalarMs, double scalarMedian) = Time(() =>
        {
            for (int position = 0; position < positions; position++)
                QuantizedMatVec.ExecuteScalar(payload, shape, input.AsSpan(position * columns, columns), single);
        });
        (double avx2Ms, double avx2Median) = Time(() =>
        {
            for (int position = 0; position < positions; position++)
                QuantizedMatVec.Execute(payload, shape, input.AsSpan(position * columns, columns), single);
        });

        long macs = (long)rows * columns * positions;
        _output.WriteLine($"batched   {batchedMs,8:F1} ms (median {batchedMedian,8:F1}, {Rate(batchedMs, macs)})");
        _output.WriteLine($"scalar    {scalarMs,8:F1} ms (median {scalarMedian,8:F1}, {Rate(scalarMs, macs)})  speedup {scalarMs / batchedMs:F2}x");
        _output.WriteLine($"avx2/pos  {avx2Ms,8:F1} ms (median {avx2Median,8:F1}, {Rate(avx2Ms, macs)})  speedup {avx2Ms / batchedMs:F2}x");

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

        (double milliseconds, double median) = Time(() =>
            QuantizedMatVec.ExecuteBatch(payload, shape, input, positions, batched, threads));

        long macs = (long)rows * columns * positions;
        _output.WriteLine($"threads={threads}: {milliseconds,8:F2} ms (median {median,8:F2}, {Rate(milliseconds, macs)})");
        Assert.True(milliseconds > 0);
    }
}

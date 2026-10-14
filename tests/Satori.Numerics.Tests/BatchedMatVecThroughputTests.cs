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
    private static (byte[] Payload, QuantizedMatrixShape Shape) Matrix(int rows, int columns) =>
        Matrix(rows, columns, BlockQuantization.Q4_K);
    /// <summary>
    /// A synthetic payload of the requested shape and format, with finite scales in the right places. The
    /// scale offsets differ per format — Q4_K keeps <c>d</c> at 0 and <c>dmin</c> at 2, Q6_K keeps its single
    /// <c>d</c> at the end of the block — so writing one at a fixed offset produced a block whose real scale
    /// was leftover random bytes. That is not a legal block and it made the format comparison meaningless.
    /// </summary>
    private static (byte[] Payload, QuantizedMatrixShape Shape) Matrix(int rows, int columns, BlockQuantization format)
    {
        var shape = new QuantizedMatrixShape(rows, columns, format);
        byte[] payload = new byte[shape.ByteLength];
        var random = new Random(20261014);
        random.NextBytes(payload);
        int blocksPerRow = columns / shape.BlockElements;
        for (int row = 0; row < rows; row++)
            for (int block = 0; block < blocksPerRow; block++)
            {
                int offset = (row * blocksPerRow + block) * shape.BlockBytes;
                switch (format)
                {
                    case BlockQuantization.Q4_K:
                        WriteHalf(payload, offset, 0.01f);
                        WriteHalf(payload, offset + 2, 0.005f);
                        break;
                    case BlockQuantization.Q6_K:
                        WriteHalf(payload, offset + Q6KCodec.BytesPerBlock - 2, 0.01f);
                        break;
                    default:
                        WriteHalf(payload, offset, 0.01f);
                        break;
                }
            }
        return (payload, shape);
    }

    private static void WriteHalf(byte[] payload, int offset, float value) =>
        BitConverter.TryWriteBytes(payload.AsSpan(offset, 2), (Half)value);

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
    /// Isolates the components of one <c>ExecuteBatch</c> call so the per-call cost can be attacked with
    /// evidence rather than assumption: the activation quantization, the row split over tasks, and the
    /// projection itself. The same bytes are used for every margin, so the only thing that changes is how the
    /// call is structured.
    /// </summary>
    [Theory]
    [InlineData("k/v", 512, 3584)]
    [InlineData("q/o", 3584, 3584)]
    public void CallCostBreakdown(string name, int rows, int columns)
    {
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(1, columns);
        float[] output = new float[rows];

        (double oneThread, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 1));
        (double twoThreads, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 2));
        (double fourThreads, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 4));
        (double eightThreads, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8));
        // Quantization alone: the same activation, quantized to Q8_K, with no projection.
        byte[] quantized = new byte[Q8KCodec.RowBytes(columns)];
        (double quantize, _) = Time(() => Q8KCodec.QuantizeRow(input, quantized));

        long macs = (long)rows * columns;
        _output.WriteLine($"{name} {rows}x{columns}, one position:");
        _output.WriteLine($"  quantize activation      {quantize * 1000,8:F1} us");
        _output.WriteLine($"  batch t=1                {oneThread * 1000,8:F1} us  ({Rate(oneThread, macs)})");
        _output.WriteLine($"  batch t=2                {twoThreads * 1000,8:F1} us  ({Rate(twoThreads, macs)})");
        _output.WriteLine($"  batch t=4                {fourThreads * 1000,8:F1} us  ({Rate(fourThreads, macs)})");
        _output.WriteLine($"  batch t=8                {eightThreads * 1000,8:F1} us  ({Rate(eightThreads, macs)})");
        _output.WriteLine($"  speedup t=8 over t=1     {oneThread / eightThreads,8:F2}x");
        Assert.True(oneThread > 0 && eightThreads > 0);
    }

    /// <summary>
    /// The decisive experiment for the gap between the kernel benchmark and the workload: the same
    /// projection, on the real artifact's bytes and on a synthetic payload of the same shape. Every figure in
    /// this class before this test is in-cache synthetic data, which answers "how fast is the kernel" and
    /// never "how fast is the model". If the real bytes run at a fraction of the synthetic ones, the
    /// difference is the memory system and the optimization belongs there.
    /// </summary>
    /// <remarks>
    /// The artifact is a 4.7 GB download kept out of version control, so the test skips when it is absent
    /// rather than failing a checkout that never fetched it. The shapes it uses are read from the artifact's
    /// own catalogue, never assumed.
    /// </remarks>
    [Fact]
    public void RealArtifactProjectionsAgainstSynthetic()
    {
        const string artifact = ".temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
        string path = Path.Combine(GgufTests.Root, artifact);
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: {artifact} is not present.");
            return;
        }

        string[] names = ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"];
        using var file = new Satori.Formats.Gguf.GgufFile(path);
        foreach (string name in names)
        {
            if (!file.Tensors.ContainsKey(name))
            {
                _output.WriteLine($"{name}: absent from the artifact");
                continue;
            }
            var matrix = new Satori.Formats.Gguf.GgufQuantizedMatrix(file, name);
            int rows = matrix.Shape.Rows, columns = matrix.Shape.Columns;
            long macs = (long)rows * columns;

            byte[] real = new byte[matrix.Shape.ByteLength];
            file.ReadTensor(name, 0, real.Length, static (bytes, target) => bytes.CopyTo(target), real);
            float[] input = Input(1, columns);
            float[] output = new float[rows];

            (double realMs, double realMedian) = Time(() =>
                QuantizedMatVec.ExecuteBatch(real, matrix.Shape, input, 1, output, threads: 8));

            var (synthetic, _) = Matrix(rows, columns, matrix.Shape.Format);
            float[] syntheticOutput = new float[rows];
            (double syntheticMs, double syntheticMedian) = Time(() =>
                QuantizedMatVec.ExecuteBatch(synthetic, matrix.Shape, input, 1, syntheticOutput, threads: 8));

            _output.WriteLine($"{name} {matrix.Shape.Format} {rows}x{columns}:");
            _output.WriteLine($"  real bytes   {realMs * 1000,9:F1} us (median {realMedian * 1000,9:F1})  {Rate(realMs, macs)}");
            _output.WriteLine($"  synthetic    {syntheticMs * 1000,9:F1} us (median {syntheticMedian * 1000,9:F1})  {Rate(syntheticMs, macs)}");
            _output.WriteLine($"  ratio        {realMs / syntheticMs,9:F2}x");
            Assert.True(realMs > 0);
        }
    }

    /// <summary>
    /// The thread curve, because the static row split makes a call take as long as its slowest worker. On a
    /// hybrid processor the workers land on cores of different speeds, so adding threads past the fast cores
    /// adds a slow worker to every call. The curve is the evidence for whether the ceiling is the core
    /// mixture, and it decides whether the thread budget should come from the topology.
    /// </summary>
    [Theory]
    [InlineData("q/o", 3584, 3584)]
    public void ThreadCurve(string name, int rows, int columns)
    {
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(1, columns);
        float[] output = new float[rows];
        long macs = (long)rows * columns;
        double best = double.MaxValue;
        int bestThreads = 0;
        _output.WriteLine($"{name} {rows}x{columns}, one position");
        for (int threads = 1; threads <= Environment.ProcessorCount; threads++)
        {
            (double milliseconds, double median) = Time(() =>
                QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads));
            if (milliseconds < best)
            {
                best = milliseconds;
                bestThreads = threads;
            }
            _output.WriteLine($"  threads={threads,3}  {milliseconds * 1000,9:F1} us (median {median * 1000,9:F1})  {Rate(milliseconds, macs)}");
        }
        _output.WriteLine($"best at {bestThreads} threads ({best * 1000:F1} us); ProcessorCount={Environment.ProcessorCount}");
        Assert.True(bestThreads >= 1);
    }

    /// <summary>
    /// Tests whether the thread pool is what caps the scaling. The batched kernel splits its rows with one
    /// <c>Task.Factory.StartNew</c> per worker and waits, and the pool injects threads at a limited rate, so a
    /// call that needs eight of them may run with fewer. Raising the pool's minimum thread count is a
    /// process-wide change that a library must not make, but as a diagnostic it separates "the pool is the
    /// wall" from "the kernel is the wall".
    /// </summary>
    [Theory]
    [InlineData("k/v", 512, 3584)]
    [InlineData("q/o", 3584, 3584)]
    public void ScalingAgainstARaisedThreadPoolFloor(string name, int rows, int columns)
    {
        var (payload, shape) = Matrix(rows, columns);
        float[] input = Input(1, columns);
        float[] output = new float[rows];

        (double before, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8));
        ThreadPool.GetMinThreads(out int minWorkers, out int minIo);
        ThreadPool.SetMinThreads(64, minIo);
        try
        {
            (double after, _) = Time(() => QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, output, threads: 8));
            _output.WriteLine($"{name} {rows}x{columns}, one position, 8 threads:");
            _output.WriteLine($"  pool default floor        {before * 1000,8:F1} us");
            _output.WriteLine($"  pool floor raised to 64   {after * 1000,8:F1} us   ratio {before / after:F2}x");
            Assert.True(after > 0);
        }
        finally
        {
            ThreadPool.SetMinThreads(minWorkers, minIo);
        }
    }

    /// <summary>
    /// The same projection called many times in a row, as a decode step does: 197 calls per token, most of
    /// them small. This is the shape of the real cost and the number a per-call optimization must move.
    /// </summary>
    [Fact]
    public void ManySmallCallsAsOneDecodeStep()
    {
        (int Rows, int Columns)[] shapes = [(3584, 3584), (512, 3584), (512, 3584), (3584, 3584),
            (18944, 3584), (18944, 3584), (3584, 18944)];
        var matrices = shapes.Select(s => Matrix(s.Rows, s.Columns)).ToArray();
        var sources = shapes.Select(s => Input(1, s.Columns)).ToArray();
        var outputs = shapes.Select(s => new float[s.Rows]).ToArray();

        // One layer: 7 projections. 28 layers is one token. Repeat the layer 28 times to time a token.
        (double perToken, _) = Time(() =>
        {
            for (int layer = 0; layer < 28; layer++)
                for (int index = 0; index < shapes.Length; index++)
                    QuantizedMatVec.ExecuteBatch(matrices[index].Payload, matrices[index].Shape,
                        sources[index], 1, outputs[index], threads: 8);
        });

        // The same work with the small k/v projections kept single-threaded, since spawning 8 tasks for
        // 512 rows may cost more than the rows do.
        (double smallSingleThread, _) = Time(() =>
        {
            for (int layer = 0; layer < 28; layer++)
                for (int index = 0; index < shapes.Length; index++)
                    QuantizedMatVec.ExecuteBatch(matrices[index].Payload, matrices[index].Shape,
                        sources[index], 1, outputs[index], threads: index is 1 or 2 ? 1 : 8);
        });

        _output.WriteLine($"196 projections, one token: {perToken:F1} ms ({1000 / perToken:F2} tok/s)");
        _output.WriteLine($"k/v single-threaded:        {smallSingleThread:F1} ms ({1000 / smallSingleThread:F2} tok/s)");
        Assert.True(perToken > 0);
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

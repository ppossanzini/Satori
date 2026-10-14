using System.Diagnostics;
using Satori.Core;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

/// <summary>
/// Holds the invariant that makes the row claim safe at any size, and carries the sweep that showed which
/// size to ship.
/// </summary>
/// <remarks>
/// The invariant is asserted always: a claim is taken once and a row writes only its own outputs, so every
/// claim size must produce identical bytes. That is an identity and does not depend on the machine.
/// <para>
/// The sweep itself measures durations and only runs when <c>SATORI_MEASURE=1</c>. It is a measurement and not
/// a gate. The first version asserted the knee inside the suite and failed on a full run with four rows at
/// 80156 us against eight at 22317 us, a 3.6x outlier that no curve produced in isolation: a descheduling
/// event, read as a property of the code. That is the failure mode rules 1 and 5 of
/// docs/measurement-method.md exist to prevent, committed by the test written to enforce them.
/// </para>
/// </remarks>
public sealed class ClaimGranularityTests
{
    private readonly ITestOutputHelper _output;

    public ClaimGranularityTests(ITestOutputHelper output) => _output = output;

    /// <summary>Set in the environment to run the duration sweeps; see the class remarks.</summary>
    private static bool Measuring => Environment.GetEnvironmentVariable("SATORI_MEASURE") == "1";

    /// <summary>Rows per claim swept, from coarser than the shipped value to finer than the knee.</summary>
    private static readonly int[] Claims = [1, 2, 4, 8, 16, 32, 64, 112, 224, 448];

    /// <summary>
    /// The claim sizes the shipped value is judged against. A claim half as fine and one twice as coarse must
    /// not beat it by more than the tolerance, so the shipped value sits in the flat region around the knee.
    /// </summary>
    private const int TolerancePercent = 10;

    /// <summary>Repetitions per measurement: one warm-up plus this many timed runs.</summary>
    private const int Repetitions = 7;

    /// <summary>Threads for every measurement, held fixed so the sweep varies one thing.</summary>
    private const int Threads = 8;

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

    private static float[] Input(int positions, int columns)
    {
        float[] values = new float[checked(positions * columns)];
        for (int index = 0; index < values.Length; index++) values[index] = (float)Math.Sin(index * 0.017) * 0.5f;
        return values;
    }

    /// <summary>A payload shaped like a real projection, with finite scales in each format's own place.</summary>
    private static (byte[] Payload, QuantizedMatrixShape Shape) Matrix(int rows, int columns, BlockQuantization format)
    {
        var shape = new QuantizedMatrixShape(rows, columns, format);
        byte[] payload = new byte[shape.ByteLength];
        new Random(20261014).NextBytes(payload);
        int blocksPerRow = columns / shape.BlockElements;
        for (int row = 0; row < rows; row++)
            for (int block = 0; block < blocksPerRow; block++)
            {
                int offset = (row * blocksPerRow + block) * shape.BlockBytes;
                int scaleOffset = format == BlockQuantization.Q6_K
                    ? offset + Q6KCodec.BytesPerBlock - 2
                    : offset;
                BitConverter.TryWriteBytes(payload.AsSpan(scaleOffset, 2), (Half)0.01f);
            }
        return (payload, shape);
    }

    /// <summary>
    /// The invariant that makes the sweep safe to run at any claim size: a claim is taken once and a row
    /// writes only its own outputs, so every granularity must produce identical bytes. Runs on a synthetic
    /// payload because it needs no artifact and asserts an identity rather than a duration.
    /// </summary>
    [Theory]
    [InlineData("q/o Q4_K", 3584, 3584, BlockQuantization.Q4_K)]
    [InlineData("down Q6_K", 3584, 18944, BlockQuantization.Q6_K)]
    public void EveryClaimSizeProducesTheSameBytes(string name, int rows, int columns, BlockQuantization format)
    {
        var (payload, shape) = Matrix(rows, columns, format);
        float[] input = Input(1, columns);
        float[] reference = new float[rows];
        QuantizedMatVec.ExecuteBatch(payload, shape, input, 1, reference, threads: Threads);
        _output.WriteLine($"{name} {rows}x{columns}: equality across {Claims.Length} claim sizes");
        foreach (int claim in Claims)
        {
            float[] output = new float[rows];
            QuantizedMatVec.ExecuteBatchTuned(payload, shape, input, 1, output, threads: Threads, rowsPerClaim: claim);
            Assert.Equal(reference, output);
        }
    }

    /// <summary>
    /// The curve on the artifact's own tensors, one square and one wide, where a call is long enough to
    /// measure. Run with <c>SATORI_MEASURE=1</c> on an otherwise idle machine; the figures in the report come
    /// from here.
    /// </summary>
    [Theory]
    [InlineData("blk.0.ffn_down.weight")]
    [InlineData("output.weight")]
    public void TheClaimIsSweptOnARealTensor(string name)
    {
        if (!Measuring)
        {
            _output.WriteLine("SKIPPED: set SATORI_MEASURE=1 to run the duration sweep.");
            return;
        }
        const string artifact = ".temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
        string path = Path.Combine(GgufTests.Root, artifact);
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: {artifact} is not present.");
            return;
        }
        using var file = new Satori.Formats.Gguf.GgufFile(path);
        if (!file.Tensors.ContainsKey(name))
        {
            _output.WriteLine($"{name}: absent from the artifact");
            return;
        }
        var matrix = new Satori.Formats.Gguf.GgufQuantizedMatrix(file, name);
        int rows = matrix.Shape.Rows, columns = matrix.Shape.Columns;
        byte[] payload = new byte[matrix.Shape.ByteLength];
        file.ReadTensor(name, 0, payload.Length, static (bytes, target) => bytes.CopyTo(target), payload);
        float[] input = Input(1, columns);
        float[] output = new float[rows];
        long macs = (long)rows * columns;

        _output.WriteLine($"{name} {matrix.Shape.Format} {rows}x{columns}, {Threads} threads, artifact bytes");
        foreach (int claim in Claims)
        {
            (double milliseconds, double median) = Time(() =>
                QuantizedMatVec.ExecuteBatchTuned(payload, matrix.Shape, input, 1, output, threads: Threads,
                    rowsPerClaim: claim));
            _output.WriteLine($"  {claim,4} rows/claim  {milliseconds * 1000,9:F1} us (median {median * 1000,9:F1})" +
                $"  {macs / milliseconds / 1e3,7:F1} GMAC/s");
        }
    }

    /// <summary>
    /// The check that makes the shipped value a shape rather than a guess: on a real tensor, a claim half as
    /// fine and one twice as coarse must not beat the shipped value by more than the tolerance. Run with
    /// <c>SATORI_MEASURE=1</c> on an idle machine; a full suite run is not an idle machine.
    /// </summary>
    [Theory]
    [InlineData("blk.0.ffn_down.weight")]
    [InlineData("output.weight")]
    public void TheShippedValueSitsAtTheKnee(string name)
    {
        if (!Measuring)
        {
            _output.WriteLine("SKIPPED: set SATORI_MEASURE=1 to run the knee check.");
            return;
        }
        const string artifact = ".temp/deepseek/dw02/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf";
        string path = Path.Combine(GgufTests.Root, artifact);
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: {artifact} is not present.");
            return;
        }
        using var file = new Satori.Formats.Gguf.GgufFile(path);
        if (!file.Tensors.ContainsKey(name))
        {
            _output.WriteLine($"{name}: absent from the artifact");
            return;
        }
        var matrix = new Satori.Formats.Gguf.GgufQuantizedMatrix(file, name);
        int rows = matrix.Shape.Rows, columns = matrix.Shape.Columns;
        byte[] payload = new byte[matrix.Shape.ByteLength];
        file.ReadTensor(name, 0, payload.Length, static (bytes, target) => bytes.CopyTo(target), payload);
        float[] input = Input(1, columns);
        float[] output = new float[rows];

        var measured = new Dictionary<int, double>();
        foreach (int claim in new[] { 4, 8, 16 })
        {
            (double milliseconds, _) = Time(() =>
                QuantizedMatVec.ExecuteBatchTuned(payload, matrix.Shape, input, 1, output, threads: Threads,
                    rowsPerClaim: claim));
            measured[claim] = milliseconds;
        }
        _output.WriteLine($"{name}: 4 rows {measured[4] * 1000:F1} us, 8 rows {measured[8] * 1000:F1} us, " +
            $"16 rows {measured[16] * 1000:F1} us");
        double allowance = measured[8] * (1 + TolerancePercent / 100.0);
        Assert.True(measured[4] <= allowance && measured[16] <= allowance,
            $"The shipped claim of 8 rows is not near the knee on {name}: 4 rows {measured[4] * 1000:F1} us, " +
            $"8 rows {measured[8] * 1000:F1} us, 16 rows {measured[16] * 1000:F1} us.");
    }

    /// <summary>
    /// A claim that cannot be served is rejected rather than silently treated as one, so a caller cannot
    /// believe it asked for something the code ignored.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void AnUnservableClaimIsRejected(int rowsPerClaim)
    {
        var (payload, shape) = Matrix(64, 512, BlockQuantization.Q4_K);
        float[] input = Input(1, 512);
        float[] output = new float[64];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuantizedMatVec.ExecuteBatchTuned(payload, shape, input, 1, output, threads: 2,
                rowsPerClaim: rowsPerClaim));
    }
}

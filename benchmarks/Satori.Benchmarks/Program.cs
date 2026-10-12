using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using Perfolizer.Horology;
using Satori.Benchmarks;
using Satori.Validation;

if (args.FirstOrDefault() == "--layouts") return TernaryLayoutBenchmarkRunner.Run(args);
if (args.FirstOrDefault() == "--gguf") return GgufBenchmarkRunner.Run(args);

bool mappedMode = args.Contains("--mapped");
args = args.Where(argument => argument != "--mapped").ToArray();
string reportDirectory = Path.GetFullPath(mappedMode ? "reports/mapped" : "reports/cpu");
Directory.CreateDirectory(reportDirectory);
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
var environment = new
{
    TimestampUtc = DateTimeOffset.UtcNow,
    Runtime = RuntimeInformation.FrameworkDescription,
    OS = RuntimeInformation.OSDescription,
    Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
    LogicalProcessors = Environment.ProcessorCount,
    Avx2 = Avx2.IsSupported,
    CorpusVersion = NumericalCorpus.Version,
    NumericalCorpus.Seed,
    Kernels = "Ternary2 scalar v1 / AVX2 v1 (scalar fallback when unavailable)",
    Mode = mappedMode
        ? "Owned versus mapped, warm, single-thread; open/validate and first-after-validation measured separately; no cold-storage claim"
        : "Warm, single-thread, in-process; quantization and setup excluded"
};
File.WriteAllText(Path.Combine(reportDirectory, "environment.json"), JsonSerializer.Serialize(environment, jsonOptions));

var numerical = new List<object>();
bool numericalPassed = true;
void ValidateSample(NumericalCase sample, int blockSize)
{
    foreach (bool useAvx2 in new[] { false, true })
    {
        var result = NumericalValidation.Evaluate(sample, blockSize, useAvx2);
        numerical.Add(new { Kernel = useAvx2 && Avx2.IsSupported ? "Avx2" : "Scalar", Result = result });
        numericalPassed &= result.Passed;
    }
}
foreach (int blockSize in new[] { 64, 128, 256 })
foreach (var sample in NumericalCorpus.CriticalCases(blockSize))
    ValidateSample(sample, blockSize);
foreach (int size in new[] { 1024, 2048, 4096 })
{
    var sample = NumericalCorpus.RandomMatrix(size, size);
    foreach (int blockSize in new[] { 64, 128, 256 })
        ValidateSample(sample, blockSize);
}
File.WriteAllText(Path.Combine(reportDirectory, "numerical.json"), JsonSerializer.Serialize(numerical, jsonOptions));
if (!numericalPassed)
{
    Console.Error.WriteLine("Numerical validation failed; benchmark skipped.");
    return 1;
}
Console.WriteLine($"Numerical corpus: {numerical.Count}/{numerical.Count} passed.");
if (mappedMode)
{
    var startup = MappedStartupProbe.Run();
    File.WriteAllText(Path.Combine(reportDirectory, "startup.json"), JsonSerializer.Serialize(startup, jsonOptions));
    if (startup.Any(result => !result.MatchesOwned))
    {
        Console.Error.WriteLine("Mapped/owned comparison failed; benchmark skipped.");
        return 1;
    }
}
if (args.Contains("--validate"))
    return 0;

var config = ManualConfig.Create(DefaultConfig.Instance)
    .WithArtifactsPath(Path.Combine(reportDirectory, "benchmarkdotnet"))
    .AddJob(Job.ShortRun
        .WithToolchain(InProcessEmitToolchain.Instance)
        .WithIterationTime(TimeInterval.FromMilliseconds(200))
        .WithId(mappedMode ? "MappedWarm" : "CpuWarm"));
var summaries = BenchmarkSwitcher.FromTypes(mappedMode
    ? [typeof(MappedTernary2Benchmarks)] : [typeof(Ternary2Benchmarks)]).Run(args, config);
var measurements = summaries.SelectMany(summary => summary.Reports).Select(report =>
{
    int size = (int)report.BenchmarkCase.Parameters.Items.Single(item => item.Name == "Size").Value;
    int blockSize = (int)report.BenchmarkCase.Parameters.Items.Single(item => item.Name == "BlockSize").Value;
    long bytes = (long)size * ((size + blockSize - 1) / blockSize) * (4 + blockSize / 4);
    double? nanoseconds = report.ResultStatistics?.Mean;
    return new
    {
        Kernel = report.BenchmarkCase.Descriptor.WorkloadMethod.Name,
        Size = size,
        BlockSize = blockSize,
        PackedBytes = bytes,
        BytesPerWeight = (double)bytes / ((long)size * size),
        MeanMicroseconds = nanoseconds / 1000,
        StandardDeviationMicroseconds = report.ResultStatistics?.StandardDeviation / 1000,
        EffectiveWeightGBPerSecond = nanoseconds > 0 ? bytes / nanoseconds : null,
        MatVecPerSecond = nanoseconds > 0 ? 1e9 / nanoseconds : null,
        AllocatedBytesPerOperation = report.GcStats.GetBytesAllocatedPerOperation(report.BenchmarkCase),
        report.Success
    };
}).ToArray();
File.WriteAllText(Path.Combine(reportDirectory, "performance.json"), JsonSerializer.Serialize(measurements, jsonOptions));
return measurements.Length > 0 && measurements.All(result => result.Success) ? 0 : 1;

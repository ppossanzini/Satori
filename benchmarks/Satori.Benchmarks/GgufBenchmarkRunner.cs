using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using Perfolizer.Horology;

namespace Satori.Benchmarks;

internal static class GgufBenchmarkRunner
{
    internal static int Run(string[] args)
    {
        if (args.Length < 2) throw new ArgumentException("--gguf requires the directory containing both stories15M files.");
        GgufBenchmarks.ModelDirectory = Path.GetFullPath(args[1]);
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .WithArtifactsPath(Path.GetFullPath("reports/gguf/benchmarkdotnet"))
            .AddExporter(JsonExporter.Full)
            .AddJob(Job.ShortRun.WithToolchain(InProcessEmitToolchain.Instance)
                .WithIterationTime(TimeInterval.FromMilliseconds(200)).WithId("GgufWarm"));
        var summaries = BenchmarkSwitcher.FromTypes([typeof(GgufBenchmarks)]).Run(args[2..], config).ToArray();
        return summaries.Length > 0 && summaries.All(s => s.Reports.All(r => r.Success)) ? 0 : 1;
    }
}

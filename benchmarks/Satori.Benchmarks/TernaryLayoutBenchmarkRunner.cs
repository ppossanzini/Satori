using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using Perfolizer.Horology;
using Satori.Validation;

namespace Satori.Benchmarks;

internal static class TernaryLayoutBenchmarkRunner
{
    internal static int Run(string[] args)
    {
        string directory = Path.GetFullPath("reports/layouts");
        Directory.CreateDirectory(directory);
        if (!TernaryLayoutValidation.Run(Path.Combine(directory, "numerical.json")))
        {
            Console.Error.WriteLine("Layout numerical validation failed; benchmark skipped.");
            return 1;
        }
        if (args.Contains("--validate")) return 0;
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .WithArtifactsPath(Path.GetFullPath("reports/layouts/benchmarkdotnet"))
            .AddExporter(JsonExporter.Full)
            .AddJob(Job.ShortRun.WithToolchain(InProcessEmitToolchain.Instance)
                .WithIterationTime(TimeInterval.FromMilliseconds(200)).WithId("LayoutsWarm"));
        var summaries = BenchmarkSwitcher.FromTypes([typeof(TernaryLayoutBenchmarks)]).Run(args[1..], config).ToArray();
        return summaries.Length > 0 && summaries.All(s => s.Reports.All(r => r.Success)) ? 0 : 1;
    }
}

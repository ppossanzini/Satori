using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json;
using Satori.Core;
using Satori.Core.Model;
using Satori.Formats.Gguf;

namespace Satori.Benchmarks;

/// <summary>
/// DS11: resident CPU baseline for a dense model. Measures startup, prefill, TTFT, decode and peak memory at
/// a chosen context, and records a small coding smoke. Writes a reproducible JSON report under
/// <c>reports/qwen/</c>.
/// </summary>
internal static class QwenBaselineRunner
{
    internal static int Run(string[] args)
    {
        if (args.Length < 3)
            throw new ArgumentException(
                "Usage: --qwen-baseline <model.gguf> <context> [--prefill N] [--decode N] [--quality N]; run from repository root.");
        string modelPath = Path.GetFullPath(args[1]);
        int context = int.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture);
        int prefill = 64, decodeTokens = 16, quality = 3;
        for (int i = 3; i + 1 < args.Length; i += 2)
        {
            int value = int.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture);
            switch (args[i])
            {
                case "--prefill": prefill = value; break;
                case "--decode": decodeTokens = value; break;
                case "--quality": quality = value; break;
                default: throw new ArgumentException($"Unknown option: {args[i]}.");
            }
        }

        string directory = Path.GetFullPath("reports/qwen");
        Directory.CreateDirectory(directory);
        using var process = Process.GetCurrentProcess();
        process.Refresh();

        // --- Startup: descriptor validation, container open, weight store, session construction ---
        long beforeManaged = GC.GetTotalAllocatedBytes(precise: true);
        long beforeWorking = process.WorkingSet64;
        long t0 = Stopwatch.GetTimestamp();
        var definition = GgufModelLoader.Load(modelPath);
        double descriptorMs = Milliseconds(t0);
        if (!definition.Capabilities.IsExecutable)
            throw new NotSupportedException($"Model is not executable: {definition.Capabilities}");
        var shape = definition.Shape!;
        t0 = Stopwatch.GetTimestamp();
        using var file = new GgufFile(modelPath);
        double openMs = Milliseconds(t0);
        t0 = Stopwatch.GetTimestamp();
        using var store = new GgufWeightStore(file);
        using var session = new DenseSession(definition, store, context);
        double sessionMs = Milliseconds(t0);
        process.Refresh();
        long startupManaged = GC.GetTotalAllocatedBytes(precise: true) - beforeManaged;

        var plan = definition.Plan(context, 16L * 1024 * 1024 * 1024);

        // --- Prefill / TTFT / decode ---
        var tokenizer = GgufBpeTokenizer.Open(modelPath);
        var tokenization = GgufBpeTokenizer.Read(modelPath);
        int[] prompt = BuildPrompt(tokenizer, prefill, context);
        var generator = new TextGenerator(session, tokenizer, tokenization.EosTokenId);

        t0 = Stopwatch.GetTimestamp();
        session.Forward(prompt, new float[shape.VocabularySize]);
        double prefillMs = Milliseconds(t0);
        int firstToken = GreedyTokenSelector.Select(session.Logits);
        double ttftMs = Milliseconds(t0);

        var decodeLatencies = new List<double>();
        int produced = 0;
        for (int i = 0; i < decodeTokens; i++)
        {
            long tick = Stopwatch.GetTimestamp();
            int next = GreedyTokenSelector.Select(session.Logits);
            session.Append(next);
            decodeLatencies.Add(Milliseconds(tick));
            produced++;
            if (tokenization.EosTokenId is int eos && next == eos) break;
        }
        double decodeTotalMs = decodeLatencies.Sum();
        decodeLatencies.Sort();
        double decodeMedianMs = decodeLatencies.Count == 0 ? 0 : decodeLatencies[decodeLatencies.Count / 2];

        process.Refresh();
        long peakWorking = process.PeakWorkingSet64;
        long peakManaged = GC.GetTotalAllocatedBytes(precise: true) - beforeManaged;

        // --- Coding smoke: greedy completion, Python snippets compiled for a reproducible signal ---
        var qualityCases = new List<object>();
        foreach (var (promptText, expected) in CodingPrompts(quality))
        {
            using var qualitySession = new DenseSession(definition, store, 512);
            var qualityGenerator = new TextGenerator(qualitySession, tokenizer, tokenization.EosTokenId);
            var result = qualityGenerator.Generate(promptText, new TextGenerationOptions(24));
            string code = promptText + result.Text;
            bool compiles = promptText.StartsWith("def ", StringComparison.Ordinal) && CompilesWithPython(code);
            qualityCases.Add(new
            {
                Prompt = promptText,
                Text = result.Text,
                Expected = expected,
                ContainsExpected = result.Text.Contains(expected, StringComparison.Ordinal),
                Compiles = compiles,
                Stop = result.Stop.ToString(),
            });
        }

        var report = new
        {
            GeneratedUtc = DateTimeOffset.UtcNow,
            Model = Path.GetFileName(modelPath),
            Context = context,
            Environment = new
            {
                Runtime = RuntimeInformation.FrameworkDescription,
                OS = RuntimeInformation.OSDescription,
                Processor = RuntimeInformation.ProcessArchitecture.ToString(),
                LogicalProcessors = Environment.ProcessorCount,
                Avx2 = Avx2.IsSupported,
                Machine = Environment.MachineName,
                TotalMemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
            },
            Shape = new
            {
                shape.Layers, shape.EmbeddingLength, shape.FeedForwardLength, shape.HeadCount,
                shape.HeadCountKv, shape.HeadDim, shape.VocabularySize, shape.TiedEmbeddings,
            },
            Plan = new
            {
                plan.WeightBytes, plan.StateBytes, plan.WorkspaceBytes, plan.TotalBytes, plan.ExceedsBudget,
                BudgetBytes = 16L * 1024 * 1024 * 1024,
            },
            Startup = new
            {
                DescriptorMs = descriptorMs,
                OpenMs = openMs,
                SessionMs = sessionMs,
                TotalMs = descriptorMs + openMs + sessionMs,
                ManagedBytes = startupManaged,
                WorkingSetBytesBefore = beforeWorking,
            },
            PromptTokenIds = prompt,
            Prefill = new { Tokens = prompt.Length, Ms = prefillMs, TokensPerSecond = prompt.Length / (prefillMs / 1000) },
            Ttft = new { Ms = ttftMs, FirstToken = firstToken },
            Decode = new
            {
                Tokens = produced,
                Ms = decodeTotalMs,
                MedianMsPerToken = decodeMedianMs,
                TokensPerSecond = produced / (decodeTotalMs / 1000),
                MaxMsPerToken = decodeLatencies.Count == 0 ? 0 : decodeLatencies[^1],
            },
            Memory = new
            {
                PeakWorkingSetBytes = peakWorking,
                PeakManagedBytes = peakManaged,
                WorkingSetAfterStartupBytes = beforeWorking,
            },
            Quality = qualityCases,
            Limits = new[]
            {
                "Prefill uses DenseSession.Forward; decode uses DenseSession.Append with the current CPU kernels.",
                "Peak working set includes the memory-mapped file pages touched, not only managed allocations.",
                "Quality is a small greedy coding smoke; the DS04 acceptance pass-rate is measured separately.",
            },
        };
        string name = $"{Path.GetFileNameWithoutExtension(modelPath)}-ctx{context}";
        File.WriteAllText(Path.Combine(directory, name + ".json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine($"{name}: startup {report.Startup.TotalMs:F1} ms; prefill {report.Prefill.Tokens} tok {report.Prefill.Ms:F1} ms "
            + $"({report.Prefill.TokensPerSecond:F2} tok/s); TTFT {report.Ttft.Ms:F1} ms; decode {report.Decode.Tokens} tok "
            + $"{report.Decode.MedianMsPerToken:F1} ms/tok ({report.Decode.TokensPerSecond:F2} tok/s); "
            + $"peak WS {report.Memory.PeakWorkingSetBytes / (1024.0 * 1024):F0} MiB; plan total {plan.TotalBytes / (1024.0 * 1024):F0} MiB "
            + $"exceeds16GiB={plan.ExceedsBudget}");
        return 0;
    }

    private static int[] BuildPrompt(Satori.Core.Tokenization.ITextTokenizer tokenizer, int tokens, int context)
    {
        tokens = Math.Clamp(tokens, 1, Math.Max(1, context - 8));
        var ids = new List<int>();
        string sentence = "public static int Add(int a, int b) => a + b;\n";
        while (ids.Count < tokens)
            ids.AddRange(tokenizer.Encode(sentence));
        return [.. ids.Take(tokens)];
    }

    private static IEnumerable<(string Prompt, string Expected)> CodingPrompts(int count)
    {
        (string, string)[] prompts =
        [
            ("def add(a, b):\n    ", "return"),
            ("def factorial(n):\n    ", "return"),
            ("def is_even(n):\n    ", "return"),
            ("public static int Add(int a, int b) =>", "a"),
        ];
        return prompts.Take(Math.Clamp(count, 0, prompts.Length));
    }

    private static bool CompilesWithPython(string code)
    {
        string path = Path.Combine(Path.GetTempPath(), $"satori-quality-{Guid.NewGuid():N}.py");
        try
        {
            File.WriteAllText(path, code, new UTF8Encoding(false));
            using var process = Process.Start(new ProcessStartInfo("python3", $"-m py_compile \"{path}\"")
            {
                RedirectStandardError = true,
                RedirectStandardOutput = true,
            });
            if (process is null) return false;
            process.WaitForExit(20_000);
            return process.HasExited && process.ExitCode == 0;
        }
        catch
        {
            return false;
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static double Milliseconds(long start) => (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
}

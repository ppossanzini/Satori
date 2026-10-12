using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text.Json;
using Satori.Core;
using Satori.Formats.Gguf;

namespace Satori.Benchmarks;

internal static class StoriesBenchmarkRunner
{
    internal static int Run(string[] args)
    {
        if (args.Length is < 2 or > 3 || (args.Length == 3 && args[2] != "--scalar"))
            throw new ArgumentException("Usage: --stories <model.gguf> [--scalar]; run from repository root.");
        var options = new StoriesBenchmarkOptions(Path.GetFullPath(args[1]),args.Length == 3);
        string directory = Path.GetFullPath(options.Scalar ? "reports/stories/scalar" : "reports/stories/auto");
        Directory.CreateDirectory(directory);
        using var process = Process.GetCurrentProcess();
        process.Refresh();
        long baselineWorkingSet = process.WorkingSet64;
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        long start = Stopwatch.GetTimestamp();
        using var model = new StoriesModel(options.ModelPath);
        double modelLoadMs = Milliseconds(start);
        long modelLoadAllocated = GC.GetAllocatedBytesForCurrentThread()-allocated;
        using var file = new GgufFile(options.ModelPath);
        start = Stopwatch.GetTimestamp();
        var tokenizer = new StoriesTokenizer(file);
        double tokenizerLoadMs = Milliseconds(start);
        process.Refresh();
        long loadedWorkingSet = process.WorkingSet64;
        using var generator = new StoriesGenerator(options.ModelPath);
        var quality = Quality(model,generator,options);
        var cases = new List<object>();
        foreach (string prompt in new[] {"", "Once upon a time", string.Join(' ',Enumerable.Repeat("a",31)), string.Join(' ',Enumerable.Repeat("a",95))})
        {
            int[] tokens = tokenizer.Encode(prompt);
            var prefill = new List<double>();
            var decode = new List<double>();
            var ttft = new List<double>();
            var complete = new List<double>();
            var allocations = new List<long>();
            var workingSets = new List<long>();
            using var session = model.CreateSession(options.Scalar);
            float[] logits = new float[StoriesModel.VocabularySize];
            for (int trial = -options.Warmups; trial < options.Samples; trial++)
            {
                session.Reset();
                start = Stopwatch.GetTimestamp();
                foreach (int token in tokens) _ = session.TryAppend(token,logits);
                int next = GreedyTokenSelector.Select(logits);
                double prefillMs = Milliseconds(start);
                start = Stopwatch.GetTimestamp();
                // Fixed work even if EOS occurs: forward throughput, not a completion stop policy.
                for (int i = 1; i < options.NewTokens; i++)
                {
                    if (!session.TryAppend(next,logits)) throw new InvalidOperationException("Benchmark context exceeded.");
                    next = GreedyTokenSelector.Select(logits);
                }
                double decodeMs = Milliseconds(start);
                double firstTextMs = double.NaN;
                allocated = GC.GetAllocatedBytesForCurrentThread();
                long generationStart = Stopwatch.GetTimestamp();
                var result = generator.Generate(prompt,new(options.NewTokens,options.Scalar),_=>
                {
                    if (double.IsNaN(firstTextMs)) firstTextMs = Milliseconds(generationStart);
                });
                double totalMs = Milliseconds(generationStart);
                long generationAllocated = GC.GetAllocatedBytesForCurrentThread()-allocated;
                if (result.Tokens.Count != options.NewTokens || result.StopReason != StoriesStopReason.TokenLimit || !double.IsFinite(firstTextMs))
                    throw new InvalidOperationException("Completion workload changed: expected twelve non-EOS tokens and emitted text.");
                process.Refresh();
                if (trial < 0) continue;
                prefill.Add(prefillMs); decode.Add(decodeMs); ttft.Add(firstTextMs); complete.Add(totalMs);
                allocations.Add(generationAllocated); workingSets.Add(process.WorkingSet64);
            }
            cases.Add(new
            {
                Prompt = prompt, PromptTokens = tokens.Length, GeneratedTokens = options.NewTokens,
                DecodeForwardTokens = options.NewTokens-1,
                PrefillMs = Statistics(prefill), DecodeMs = Statistics(decode), FirstTextMs = Statistics(ttft), CompletionMs = Statistics(complete),
                PrefillTokensPerSecond = tokens.Length*1000/prefill.Average(),
                DecodeTokensPerSecond = (options.NewTokens-1)*1000/decode.Average(),
                CompletionAllocatedBytes = allocations, WorkingSetBytes = workingSets,
                PrefillSamplesMs = prefill, DecodeSamplesMs = decode, FirstTextSamplesMs = ttft, CompletionSamplesMs = complete
            });
            Console.WriteLine($"prompt={tokens.Length}; TTFT={ttft.Average():F3} ms; prefill={prefill.Average():F3} ms; decode={(options.NewTokens-1)*1000/decode.Average():F2} token/s");
        }
        process.Refresh();
        var report = new
        {
            TimestampUtc = DateTimeOffset.UtcNow, Options = options, Sampling = new SamplingOptions(),
            StopwatchFrequency = Stopwatch.Frequency,
            TieredPgoOverride = Environment.GetEnvironmentVariable("DOTNET_TieredPGO"),
            Avx2Override = Environment.GetEnvironmentVariable("DOTNET_EnableAVX2"),
            Runtime = RuntimeInformation.FrameworkDescription, OS = RuntimeInformation.OSDescription,
            Architecture = RuntimeInformation.ProcessArchitecture.ToString(), LogicalProcessors = Environment.ProcessorCount,
            Cpu = File.Exists("/proc/cpuinfo") ? File.ReadLines("/proc/cpuinfo").FirstOrDefault(l=>l.StartsWith("model name",StringComparison.Ordinal)) : null,
            Avx2 = Avx2.IsSupported, Kernel = options.Scalar || !Avx2.IsSupported ? "scalar" : "AVX2",
            SourceSha256 = SourceHash(), ModelSha256 = Hash(File.ReadAllBytes(options.ModelPath)),
            ModelBytes = new FileInfo(options.ModelPath).Length, ContextCapacity = StoriesModel.ContextLength,
            KvPayloadBytesPerSession = StoriesSession.CacheByteLength,
            ModelLoadMs = modelLoadMs, TokenizerLoadMs = tokenizerLoadMs, ModelLoadAllocatedBytes = modelLoadAllocated,
            BaselineWorkingSetBytes = baselineWorkingSet, LoadedWorkingSetBytes = loadedWorkingSet,
            PeakWorkingSetBytes = process.PeakWorkingSet64, ManagedHeapBytesAtEnd = GC.GetTotalMemory(false),
            MemoryScope = "Process contains a phase model and a separate generator model, plus tokenizer, validation and report data. Working set includes shared mapped pages and runtime; peak is OS process high-water mark, not model-only memory.",
            TimingScope = "Warm single-thread sequential CPU; 3 warmups and 15 samples per prompt; model load is first in process with uncontrolled page cache. End-to-end includes tokenizer, session/sampler allocation, decode, callback and disposal. Phase timings exclude session allocation/reset/tokenization.",
            Quality = quality, Cases = cases
        };
        File.WriteAllText(Path.Combine(directory,"report.json"),JsonSerializer.Serialize(report,new JsonSerializerOptions {WriteIndented=true}));
        return 0;
    }

    private static double Milliseconds(long start) => Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    private static object Statistics(List<double> values)
    {
        double[] sorted = values.Order().ToArray();
        double mean = values.Average();
        return new { Mean = mean, Median = sorted[sorted.Length/2], Min = sorted[0], Max = sorted[^1],
            P95 = sorted[(int)Math.Ceiling(sorted.Length*0.95)-1],
            StandardDeviation = Math.Sqrt(values.Sum(v=>(v-mean)*(v-mean))/(values.Count-1)) };
    }

    private static object Quality(StoriesModel model, StoriesGenerator generator, StoriesBenchmarkOptions options)
    {
        const string root = "fixtures/models/stories15m-q8-v1";
        using var numpy = JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"numpy/reference.json")));
        using var official = JsonDocument.Parse(File.ReadAllText(Path.Combine(root,"official-f32/reference.json")));
        var results = new List<object>();
        foreach (var item in numpy.RootElement.GetProperty("forward_cases").EnumerateArray())
        {
            string id = item.GetProperty("id").GetString()!;
            int[] tokens = item.GetProperty("tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
            var record = item.GetProperty("logits");
            byte[] expected = File.ReadAllBytes(Path.Combine(root,"numpy",record.GetProperty("file").GetString()!));
            if (Hash(expected) != record.GetProperty("sha256").GetString() || expected.Length != StoriesModel.VocabularySize*8)
                throw new InvalidDataException("Reference logits hash or size mismatch.");
            float[] logits = new float[StoriesModel.VocabularySize];
            model.Forward(tokens,logits,options.Scalar);
            double maxError = 0, squared = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                double reference = BinaryPrimitives.ReadDoubleLittleEndian(expected.AsSpan(i*8,8));
                double error = Math.Abs(logits[i]-reference);
                if (!float.IsFinite(logits[i]) || error > 5e-4+5e-5*Math.Abs(reference)) throw new InvalidDataException("Logits diverged from reference.");
                maxError = Math.Max(maxError,error); squared += error*error;
            }
            double rms = Math.Sqrt(squared/logits.Length);
            if (rms > 1e-4) throw new InvalidDataException("Logits RMS exceeded reference gate.");
            string prompt = official.RootElement.GetProperty("tokenizer_cases").EnumerateArray().First(v=>v.GetProperty("id").GetString()==id).GetProperty("text").GetString()!;
            int[] referenceTokens = official.RootElement.GetProperty("forward_cases").EnumerateArray().Single(v=>v.GetProperty("id").GetString()==id).GetProperty("greedy_tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
            var generated = generator.Generate(prompt,new(options.NewTokens,options.Scalar));
            if (!generated.Tokens.SequenceEqual(referenceTokens)) throw new InvalidDataException("Greedy sequence differs from official reference.");
            results.Add(new {Id=id, MaxAbsoluteError=maxError, RmsError=rms, MatchingGreedyTokens=referenceTokens.Length, generated.Text});
        }
        return results;
    }

    private static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string SourceHash()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (string path in Directory.EnumerateFiles("src","*.cs",SearchOption.AllDirectories)
            .Concat(Directory.EnumerateFiles("benchmarks/Satori.Benchmarks","*.cs",SearchOption.AllDirectories))
            .Concat(Directory.EnumerateFiles("src","*.csproj",SearchOption.AllDirectories))
            .Concat(new[] {"Directory.Build.props","global.json","benchmarks/Satori.Benchmarks/Satori.Benchmarks.csproj"})
            .Where(p=>!p.Split(Path.DirectorySeparatorChar).Any(s=>s is "bin" or "obj")).Order(StringComparer.Ordinal))
        {
            hash.AppendData(System.Text.Encoding.UTF8.GetBytes(path+"\n"));
            hash.AppendData(File.ReadAllBytes(path));
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
}

using Satori.Core;
using Satori.Core.Model;
using Satori.Core.Tokenization;
using System.Runtime.Intrinsics.X86;
using Satori.Formats.Gguf;
using System.Globalization;

if (args.Length > 0)
{
    try
    {
        if (args[0] == "inventory")
        {
            if (args.Length is < 2 or > 3 || (args.Length == 3 && args[2] != "--json"))
            {
                Console.Error.WriteLine("Usage: Satori.Console inventory <file.gguf> [--json]");
                return 1;
            }
            var inventory = GgufInventory.Scan(args[1]);
            Console.Write(args.Length == 3 ? inventory.ToJson() : inventory.ToReport());
            return 0;
        }
        if (args[0] == "describe")
        {
            if (args.Length is < 2 or > 3)
            {
                Console.Error.WriteLine("Usage: Satori.Console describe <model.gguf> [context-length]");
                return 1;
            }
            var definition = GgufModelLoader.Load(args[1]);
            int context = args.Length == 3 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 2048;
            Console.WriteLine($"Architecture: {definition.Architecture}");
            Console.WriteLine($"Tensors: {definition.Tensors.Count}; compressed weight bytes: {definition.WeightBytes}");
            Console.WriteLine($"Executable: {(definition.Capabilities.IsExecutable ? "yes" : "no")}");
            foreach (var gap in definition.Capabilities.Gaps)
                Console.WriteLine($"  gap [{gap.Kind}] {gap.Detail}: {gap.Reason}");
            if (definition.Shape is { } shape)
            {
                Console.WriteLine($"Shape: layers={shape.Layers} embedding={shape.EmbeddingLength} ffn={shape.FeedForwardLength} heads={shape.HeadCount}/{shape.HeadCountKv} head_dim={shape.HeadDim} vocab={shape.VocabularySize} tied={shape.TiedEmbeddings}");
                long budget = 12L * 1024 * 1024 * 1024;
                var plan = definition.Plan(context, budget);
                Console.WriteLine($"Plan @ {context}: weights={plan.WeightBytes} state={plan.StateBytes} workspace={plan.WorkspaceBytes} total={plan.TotalBytes} exceeds_budget={plan.ExceedsBudget}");
            }
            else
            {
                Console.WriteLine("Shape: not derived (see gaps)");
            }
            return 0;
        }
        if (args[0] == "text")
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("Usage: Satori.Console text <model.gguf> <prompt> [max-new-tokens] [--context N] [--threads N] [--budget BYTES] [--temperature N] [--top-k N] [--top-p N] [--seed N]");
                return 1;
            }
            var options = new TextGenerationOptions(16);
            var sampling = new SamplingOptions();
            int context = 2048;
            int threads = InferenceOptions.Default.Threads;
            long budget = 0;
            int cursor = 3;
            if (cursor < args.Length && !args[cursor].StartsWith("--", StringComparison.Ordinal))
            {
                options = options with { MaxNewTokens = int.Parse(args[cursor++], NumberStyles.None, CultureInfo.InvariantCulture) };
            }
            while (cursor < args.Length)
            {
                string flag = args[cursor++];
                if (cursor == args.Length) throw new ArgumentException($"Missing value for {flag}.");
                string value = args[cursor++];
                if (flag == "--context")
                {
                    context = int.Parse(value, CultureInfo.InvariantCulture);
                    continue;
                }
                if (flag == "--threads")
                {
                    threads = int.Parse(value, CultureInfo.InvariantCulture);
                    continue;
                }
                if (flag == "--budget")
                {
                    budget = ParseByteSize(value);
                    continue;
                }
                sampling = flag switch
                {
                    "--temperature" => sampling with { Temperature = double.Parse(value, CultureInfo.InvariantCulture) },
                    "--top-k" => sampling with { TopK = int.Parse(value, CultureInfo.InvariantCulture) },
                    "--top-p" => sampling with { TopP = double.Parse(value, CultureInfo.InvariantCulture) },
                    "--seed" => sampling with { Seed = ulong.Parse(value, CultureInfo.InvariantCulture) },
                    _ => throw new ArgumentException($"Unknown option: {flag}.")
                };
            }
            sampling.Validate();
            options = options with { Sampling = sampling };

            var definition = GgufModelLoader.Load(args[1]);
            if (!definition.Capabilities.IsExecutable)
                throw new NotSupportedException($"Model is not executable: {definition.Capabilities}");
            var tokenization = GgufBpeTokenizer.Read(args[1]);
            var tokenizer = GgufBpeTokenizer.Open(args[1]);
            using var file = new GgufFile(args[1]);
            using var store = new GgufWeightStore(file);
            using var session = new DenseSession(definition, store, context,
                options: new InferenceOptions(threads) { BudgetBytes = budget });
            var generator = new TextGenerator(session, tokenizer, tokenization.EosTokenId);

            // Ctrl+C asks for a stop rather than killing the process, so the latency the DW06.8 criterion
            // bounds is observable instead of being the operating system's kill. The stamp is taken in the
            // handler and reported by the catch, which is the interval the criterion is about.
            using var cancellation = new CancellationTokenSource();
            long requestedAt = 0;
            ConsoleCancelEventHandler handler = (_, eventArgs) =>
            {
                eventArgs.Cancel = true;
                Interlocked.CompareExchange(ref requestedAt, Environment.TickCount64, 0);
                cancellation.Cancel();
            };
            Console.CancelKeyPress += handler;
            try
            {
                long started = Environment.TickCount64;
                var generation = generator.Generate(args[2], options, Console.Write, cancellation.Token);
                Console.WriteLine();
                Console.Error.WriteLine($"{generation.Stop}; prompt: {generation.PromptTokens.Count}; generated: {generation.GeneratedTokens.Count}; " +
                    $"context: {context}; budget: {budget}; elapsed_ms: {Environment.TickCount64 - started}");
                return 0;
            }
            catch (OperationCanceledException)
            {
                long waited = requestedAt == 0 ? 0 : Environment.TickCount64 - requestedAt;
                Console.Error.WriteLine($"cancelled; positions: {session.Position}; cancel_latency_ms: {waited}");
                return 130;
            }
            finally
            {
                Console.CancelKeyPress -= handler;
            }
        }
        if (args.Length != 2 || args[0] is not ("save" or "run"))
        {
            Console.Error.WriteLine("Usage: Satori.Console [save <new-file> | run <file> | inventory <file.gguf> [--json] | describe <model.gguf> [context] | text <model.gguf> <prompt> [max-new-tokens]]");
            return 1;
        }
        if (args[0] == "save")
        {
            Ternary2Matrix.Quantize([2, -2, 0, 0, 1, -1], 2, 3).Save(args[1]);
            Console.WriteLine($"Saved Ternary2 demo: {args[1]}");
            return 0;
        }
        using var mapped = new MappedTernary2Matrix(args[1]);
        float[] mappedInput = Enumerable.Repeat(1f, mapped.Columns).ToArray();
        if (mapped.Columns == 3) mappedInput = [2, -1, 3];
        float[] mappedOutput = new float[mapped.Rows];
        mapped.Execute(mappedInput, mappedOutput);
        Console.WriteLine($"Mapped Ternary2 {(Avx2.IsSupported ? "AVX2" : "scalar")}: [{string.Join(", ", mappedOutput.Take(8))}]");
        Console.WriteLine($"Shape: {mapped.Rows} x {mapped.Columns}; payload: {mapped.ByteLength} bytes");
        return 0;
    }
    catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or NotSupportedException or FormatException or OverflowException or InvalidOperationException)
    {
        Console.Error.WriteLine(exception.Message);
        return 1;
    }
}

float[] weights = [2, -2, 0, 0, 1, -1];
float[] input = [2, -1, 3];
var matrix = Ternary2Matrix.Quantize(weights, 2, 3);
float[] output = new float[2];
Ternary2MatVec.Execute(matrix, input, output);
Console.WriteLine($"Ternary2 {(Avx2.IsSupported ? "AVX2" : "scalar")}: [{string.Join(", ", output)}]");
Console.WriteLine($"Packed weights: {matrix.ByteLength} bytes (including scales and row padding)");
return 0;

/// <summary>
/// Parses a byte size as a plain count or with a K, M, G or T suffix, so the budget can be stated the way an
/// operator thinks about memory. A suffix is a power of 1024, and the value is checked for overflow rather
/// than silently wrapping.
/// </summary>
static long ParseByteSize(string value)
{
    if (string.IsNullOrEmpty(value)) throw new ArgumentException("Empty byte size.");
    long multiplier = 1;
    ReadOnlySpan<char> span = value.AsSpan();
    switch (char.ToUpperInvariant(span[^1]))
    {
        case 'K': multiplier = 1L << 10; span = span[..^1]; break;
        case 'M': multiplier = 1L << 20; span = span[..^1]; break;
        case 'G': multiplier = 1L << 30; span = span[..^1]; break;
        case 'T': multiplier = 1L << 40; span = span[..^1]; break;
    }
    long count = long.Parse(span, NumberStyles.None, CultureInfo.InvariantCulture);
    return checked(count * multiplier);
}

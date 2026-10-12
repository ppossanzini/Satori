using Satori.Core;
using System.Runtime.Intrinsics.X86;
using Satori.Formats.Gguf;
using System.Globalization;

if (args.Length > 0)
{
    try
    {
        if (args[0] == "generate")
        {
            if (args.Length < 3)
            {
                Console.Error.WriteLine("Usage: Satori.Console generate <model.gguf> <prompt> [max-new-tokens] [--temperature N] [--top-k N] [--top-p N] [--seed N]");
                return 1;
            }
            var options = new StoriesGenerationOptions();
            var sampling = new SamplingOptions();
            int cursor = 3;
            if (cursor < args.Length && !args[cursor].StartsWith("--",StringComparison.Ordinal))
            {
                options = options with { MaxNewTokens = int.Parse(args[cursor++],NumberStyles.None,CultureInfo.InvariantCulture) };
            }
            while (cursor < args.Length)
            {
                string flag = args[cursor++];
                if (cursor == args.Length) throw new ArgumentException($"Missing value for {flag}.");
                string value = args[cursor++];
                sampling = flag switch
                {
                    "--temperature" => sampling with { Temperature = double.Parse(value,CultureInfo.InvariantCulture) },
                    "--top-k" => sampling with { TopK = int.Parse(value,CultureInfo.InvariantCulture) },
                    "--top-p" => sampling with { TopP = double.Parse(value,CultureInfo.InvariantCulture) },
                    "--seed" => sampling with { Seed = ulong.Parse(value,CultureInfo.InvariantCulture) },
                    _ => throw new ArgumentException($"Unknown option: {flag}.")
                };
            }
            sampling.Validate();
            options = options with { Sampling = sampling };
            using var generator = new StoriesGenerator(args[1]);
            var result = generator.Generate(args[2],options,Console.Write);
            Console.WriteLine();
            Console.Error.WriteLine($"{result.StopReason}; prompt: {result.PromptTokenCount}; generated: {result.Tokens.Count}");
            return 0;
        }
        if (args.Length != 2 || args[0] is not ("save" or "run"))
        {
            Console.Error.WriteLine("Usage: Satori.Console [save <new-file> | run <file> | generate <model.gguf> <prompt> [max-new-tokens]]");
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
    catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or NotSupportedException or FormatException or OverflowException)
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

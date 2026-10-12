using System.Diagnostics;
using System.Text.Json;
using System.Security.Cryptography;
using Satori.Formats.Gguf;

if (args.Length != 2)
{
    Console.Error.WriteLine("Usage: Satori.Gguf.Validation <model.gguf> <independent-reference.json>");
    return 2;
}
using var reference = JsonDocument.Parse(File.ReadAllText(args[1]));
using (var stream = File.OpenRead(args[0]))
{
    string digest = Convert.ToHexStringLower(SHA256.HashData(stream));
    if (digest != reference.RootElement.GetProperty("sha256").GetString())
        throw new InvalidDataException("Model checksum differs from the independent reference.");
}
using var file = new GgufFile(args[0]);
if (file.Tensors.Count != reference.RootElement.GetProperty("catalogue").GetArrayLength())
    throw new InvalidDataException("Catalogue tensor count differs from official reader.");
foreach (var item in reference.RootElement.GetProperty("catalogue").EnumerateArray())
{
    var actual = file.Tensors[item.GetProperty("name").GetString()!];
    if (actual.Offset != item.GetProperty("offset").GetInt64() || actual.ByteLength != item.GetProperty("bytes").GetInt64() ||
        actual.Type != item.GetProperty("type").GetUInt32() ||
        !actual.Dimensions.SequenceEqual(item.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64())))
        throw new InvalidDataException("Catalogue differs from official reader.");
}
var results = new List<object>();
bool passed = true;
foreach (var entry in reference.RootElement.GetProperty("tensors").EnumerateArray())
{
    string name = entry.GetProperty("name").GetString()!;
    var matrix = new GgufQuantizedMatrix(file, name);
    int columns = matrix.Shape.Columns;
    float[] input = Enumerable.Range(0, columns).Select(i => (i * 17 % 31 - 15) / 16f).ToArray();
    float[] output = new float[matrix.Shape.Rows];
    matrix.Execute(input, output);
    double maximumRatio = 0;
    var expected = entry.GetProperty("output").EnumerateArray().Select(x => x.GetDouble()).ToArray();
    var magnitude = entry.GetProperty("sum_abs").EnumerateArray().Select(x => x.GetDouble()).ToArray();
    if (expected.Length != output.Length) throw new InvalidDataException("Reference shape mismatch.");
    double n = columns + columns / 32.0 + 2;
    double gamma = n * Math.ScaleB(1.0, -24) / (1 - n * Math.ScaleB(1.0, -24));
    for (int r = 0; r < output.Length; r++)
    {
        double bound = gamma * magnitude[r] + n * float.Epsilon;
        double error = Math.Abs(output[r] - expected[r]);
        maximumRatio = Math.Max(maximumRatio, error / Math.Max(bound, double.Epsilon));
        passed &= double.IsFinite(error) && error <= bound;
    }
    for (int i = 0; i < 5; i++) matrix.Execute(input, output);
    long allocated = GC.GetAllocatedBytesForCurrentThread();
    long start = Stopwatch.GetTimestamp();
    for (int i = 0; i < 20; i++) matrix.Execute(input, output);
    double microseconds = Stopwatch.GetElapsedTime(start).TotalMicroseconds / 20;
    allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
    passed &= allocated == 0;
    results.Add(new { Name = name, matrix.Shape.Rows, columns, matrix.Shape.Format, MaximumErrorOverBound = maximumRatio,
        AllocatedBytesPerOperation = allocated / 20, AutoMicroseconds = microseconds });
}
Console.WriteLine(JsonSerializer.Serialize(new { Passed = passed, TensorCount = file.Tensors.Count, Results = results },
    new JsonSerializerOptions { WriteIndented = true }));
return passed && results.Count > 0 ? 0 : 1;

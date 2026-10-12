namespace Satori.Validation;

public static class NumericalCorpus
{
    public const uint Seed = 1729;
    public const int Version = 1;

    public static NumericalCase RandomMatrix(int rows, int columns)
    {
        uint state = Seed;
        float[] weights = new float[checked(rows * columns)];
        float[] input = new float[columns];
        for (int i = 0; i < weights.Length; i++) weights[i] = Next(ref state) * 2;
        for (int i = 0; i < input.Length; i++) input[i] = Next(ref state);
        return new("random", rows, columns, weights, input);
    }

    public static IEnumerable<NumericalCase> CriticalCases(int blockSize)
    {
        yield return RandomMatrix(7, blockSize * 2 + 3);
        yield return Pattern("zeros", 3, blockSize + 1, _ => 0, _ => 1);
        yield return Pattern("alternating", 5, blockSize * 2 + 1, i => i % 2 == 0 ? 1 : -1, _ => 1);
        yield return Pattern("sparse", 3, blockSize - 1, i => i % 17 == 0 ? 4 : 0, i => i % 3 - 1);
        yield return Pattern("small-scale", 3, blockSize + 3, i => i % 2 == 0 ? 1e-30f : -1e-30f, _ => 1);
        yield return Pattern("large-scale", 3, blockSize + 3, i => i % 2 == 0 ? 1e30f : -1e30f, _ => 1e-10f);
        yield return Pattern("subnormal", 2, blockSize + 1, i => i % 2 == 0 ? float.Epsilon : -float.Epsilon, _ => 1);
        yield return Pattern("max-scale", 2, blockSize + 1, i => i % 2 == 0 ? float.MaxValue : -float.MaxValue, _ => 1e-38f);
        yield return Pattern("single-weight", 1, 1, _ => -2, _ => 3);
    }

    private static NumericalCase Pattern(string name, int rows, int columns, Func<int, float> weight, Func<int, float> activation)
    {
        float[] weights = new float[rows * columns];
        float[] input = new float[columns];
        for (int i = 0; i < weights.Length; i++) weights[i] = weight(i);
        for (int i = 0; i < input.Length; i++) input[i] = activation(i);
        return new(name, rows, columns, weights, input);
    }

    // Xorshift32 and a fixed 24-bit conversion keep corpus bytes stable across runtimes.
    private static float Next(ref uint state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (state >> 8) * (1f / 8388608f) - 1;
    }
}

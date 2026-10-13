namespace Satori.Core;

/// <summary>Allocation-free FP32 operators. Inputs must be finite; dimensions are exact.
/// Elementwise operators permit exact in-place output, but reject partial overlap.</summary>
public static class TransformerOps
{
    private static void Output(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != output.Length) throw new ArgumentException("Lengths differ.");
        if (input.Overlaps(output, out int offset) && offset != 0) throw new ArgumentException("Partial overlap.");
    }

    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, float epsilon, Span<float> output)
    {
        Output(input, output);
        Output(weights, output);
        if (input.IsEmpty || !float.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        double sum = 0;
        foreach (float value in input) sum += (double)value * value;
        double scale = 1 / Math.Sqrt(sum / input.Length + epsilon);
        for (int i = 0; i < input.Length; i++) output[i] = (float)(input[i] * scale * weights[i]);
    }

    public static void Rope(Span<float> vector, int headDimension, int position)
    {
        if (headDimension <= 0 || headDimension % 2 != 0 || vector.Length % headDimension != 0 || position < 0)
            throw new ArgumentOutOfRangeException(nameof(headDimension));
        for (int j = 0; j < headDimension; j += 2)
        {
            double angle = position / Math.Pow(10000, (double)j / headDimension);
            var (sin, cos) = Math.SinCos(angle);
            for (int start = 0; start < vector.Length; start += headDimension)
            {
                double x = vector[start + j], y = vector[start + j + 1];
                vector[start + j] = (float)(x * cos - y * sin);
                vector[start + j + 1] = (float)(x * sin + y * cos);
            }
        }
    }

    public static void SwiGlu(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> output)
    {
        Output(gate, output);
        Output(up, output);
        for (int i = 0; i < gate.Length; i++)
        {
            double x = gate[i];
            double sigmoid = x >= 0 ? 1 / (1 + Math.Exp(-x)) : Math.Exp(x) / (1 + Math.Exp(x));
            output[i] = (float)(x * sigmoid * up[i]);
        }
    }

    public static void Add(ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> output)
    {
        Output(left, output);
        Output(right, output);
        for (int i = 0; i < left.Length; i++) output[i] = left[i] + right[i];
    }

    /// <summary>Token-major K/V prefix through the current token, with equal Q/K/V heads.
    /// Caller owns the cache; future positions must not be included in the prefix.</summary>
    public static void Attention(ReadOnlySpan<float> query, ReadOnlySpan<float> keys, ReadOnlySpan<float> values,
        int headDimension, int tokenCount, Span<double> scores, Span<float> output)
    {
        if (headDimension <= 0 || query.IsEmpty || query.Length % headDimension != 0 || tokenCount <= 0 ||
            keys.Length != checked(tokenCount * query.Length) || values.Length != keys.Length || scores.Length < tokenCount || output.Length != query.Length)
            throw new ArgumentException("Invalid attention dimensions.");
        if (query.Overlaps(output) || keys.Overlaps(output) || values.Overlaps(output)) throw new ArgumentException("Attention output overlaps input.");
        if (System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores).Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(query)) ||
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores).Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(keys)) ||
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores).Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values)) ||
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(scores).Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Scratch overlaps input or output.");
        double scale = 1 / Math.Sqrt(headDimension);
        for (int head = 0; head < query.Length; head += headDimension)
        {
            double max = double.NegativeInfinity;
            for (int token = 0; token < tokenCount; token++)
            {
                double dot = 0;
                for (int j = 0; j < headDimension; j++) dot += (double)query[head + j] * keys[token * query.Length + head + j];
                scores[token] = dot * scale;
                max = Math.Max(max, scores[token]);
            }
            double total = 0;
            for (int token = 0; token < tokenCount; token++) total += scores[token] = Math.Exp(scores[token] - max);
            for (int j = 0; j < headDimension; j++)
            {
                double sum = 0;
                for (int token = 0; token < tokenCount; token++) sum += scores[token] / total * values[token * query.Length + head + j];
                output[head + j] = (float)sum;
            }
        }
    }

    /// <summary>Rotary embedding that splits each head in half (GPT-NeoX convention): the pair
    /// <c>(k, k + half)</c> is rotated with <c>angle = position / base^(2k / rotaryDimensions)</c>. Only the
    /// first <paramref name="rotaryDimensions"/> lanes of each head are rotated; the rest pass through.</summary>
    public static void RopeHalf(Span<float> vector, int headDimension, int position, double frequencyBase, int rotaryDimensions)
    {
        if (headDimension <= 0 || vector.Length % headDimension != 0 || position < 0)
            throw new ArgumentOutOfRangeException(nameof(headDimension));
        if (!double.IsFinite(frequencyBase) || frequencyBase <= 0) throw new ArgumentOutOfRangeException(nameof(frequencyBase));
        if (rotaryDimensions <= 0 || rotaryDimensions > headDimension || rotaryDimensions % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDimensions));
        int half = rotaryDimensions / 2;
        for (int k = 0; k < half; k++)
        {
            double angle = position / Math.Pow(frequencyBase, 2.0 * k / rotaryDimensions);
            var (sin, cos) = Math.SinCos(angle);
            for (int start = 0; start < vector.Length; start += headDimension)
            {
                double x = vector[start + k], y = vector[start + k + half];
                vector[start + k] = (float)(x * cos - y * sin);
                vector[start + k + half] = (float)(x * sin + y * cos);
            }
        }
    }

    /// <summary>RMS normalization applied independently to each head, using a per-lane weight of length
    /// <paramref name="headDimension"/>. Permits exact in-place use.</summary>
    public static void RmsNormHeads(Span<float> vector, ReadOnlySpan<float> weights, float epsilon, int headDimension)
    {
        if (headDimension <= 0 || vector.Length % headDimension != 0)
            throw new ArgumentOutOfRangeException(nameof(headDimension));
        if (weights.Length != headDimension) throw new ArgumentException("Per-head weights must match the head dimension.");
        if (!float.IsFinite(epsilon) || epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon));
        for (int start = 0; start < vector.Length; start += headDimension)
        {
            double sum = 0;
            for (int i = 0; i < headDimension; i++) sum += (double)vector[start + i] * vector[start + i];
            double scale = 1 / Math.Sqrt(sum / headDimension + epsilon);
            for (int i = 0; i < headDimension; i++) vector[start + i] = (float)(vector[start + i] * scale * weights[i]);
        }
    }

    /// <summary>Grouped-query attention. Query has <paramref name="queryHeads"/> heads; keys and values are
    /// token-major with <paramref name="kvHeads"/> heads. Query head <c>h</c> attends to KV head
    /// <c>h / (queryHeads / kvHeads)</c>. Causal prefix through the current token; caller owns the cache.</summary>
    public static void GroupedAttention(ReadOnlySpan<float> query, ReadOnlySpan<float> keys, ReadOnlySpan<float> values,
        int queryHeads, int kvHeads, int headDimension, int tokenCount, Span<double> scores, Span<float> output)
    {
        if (headDimension <= 0 || queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0 || tokenCount <= 0)
            throw new ArgumentException("Invalid grouped-attention dimensions.");
        int queryWidth = checked(queryHeads * headDimension);
        int kvWidth = checked(kvHeads * headDimension);
        if (query.Length != queryWidth || output.Length != queryWidth)
            throw new ArgumentException("Query or output width does not match the head count.");
        if (keys.Length != checked(tokenCount * kvWidth) || values.Length != keys.Length)
            throw new ArgumentException("Key/value width does not match the KV head count.");
        if (scores.Length < tokenCount) throw new ArgumentException("Score scratch is too small.");
        if (query.Overlaps(output) || keys.Overlaps(output) || values.Overlaps(output))
            throw new ArgumentException("Attention output overlaps input.");
        int groupSize = queryHeads / kvHeads;
        double scale = 1 / Math.Sqrt(headDimension);
        for (int head = 0; head < queryHeads; head++)
        {
            int kvHead = head / groupSize;
            int queryOffset = head * headDimension;
            double max = double.NegativeInfinity;
            for (int token = 0; token < tokenCount; token++)
            {
                int keyOffset = token * kvWidth + kvHead * headDimension;
                double dot = 0;
                for (int j = 0; j < headDimension; j++) dot += (double)query[queryOffset + j] * keys[keyOffset + j];
                scores[token] = dot * scale;
                max = Math.Max(max, scores[token]);
            }
            double total = 0;
            for (int token = 0; token < tokenCount; token++) total += scores[token] = Math.Exp(scores[token] - max);
            for (int j = 0; j < headDimension; j++)
            {
                double sum = 0;
                for (int token = 0; token < tokenCount; token++)
                    sum += scores[token] / total * values[token * kvWidth + kvHead * headDimension + j];
                output[queryOffset + j] = (float)sum;
            }
        }
    }
}

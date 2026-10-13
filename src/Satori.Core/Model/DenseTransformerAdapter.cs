namespace Satori.Core.Model;

/// <summary>
/// Dense decoder with grouped-query attention. Covers the qwen2 and qwen3 families: qwen3 adds Q/K per-head
/// normalization and declares an explicit key/value length that is honored instead of
/// <c>EmbeddingLength / HeadCount</c>.
/// </summary>
public sealed class DenseTransformerAdapter : IArchitectureAdapter
{
    public IReadOnlyList<string> ArchitectureIds { get; } = ["qwen2", "qwen3"];

    public IReadOnlyList<CapabilityGap> Validate(
        ModelConfig config, IReadOnlyDictionary<string, ModelTensor> tensors, out ModelShape? shape)
    {
        var gaps = new List<CapabilityGap>();
        shape = null;

        if (!config.TryInt("block_count", out long layers) || layers <= 0)
            gaps.Add(new("architecture", "block_count", "missing or non-positive"));
        if (!config.TryInt("embedding_length", out long embedding) || embedding <= 0)
            gaps.Add(new("architecture", "embedding_length", "missing or non-positive"));
        if (!config.TryInt("feed_forward_length", out long feedForward) || feedForward <= 0)
            gaps.Add(new("architecture", "feed_forward_length", "missing or non-positive"));
        if (!config.TryInt("attention.head_count", out long heads) || heads <= 0)
            gaps.Add(new("architecture", "attention.head_count", "missing or non-positive"));
        if (!config.TryInt("attention.head_count_kv", out long headsKv) || headsKv <= 0)
            gaps.Add(new("architecture", "attention.head_count_kv", "missing or non-positive"));
        if (!config.TryFloat("attention.layer_norm_rms_epsilon", out double epsilon) || epsilon <= 0)
            gaps.Add(new("architecture", "attention.layer_norm_rms_epsilon", "missing or non-positive"));
        if (!config.TryFloat("rope.freq_base", out double rope) || rope <= 0)
            gaps.Add(new("architecture", "rope.freq_base", "missing or non-positive"));
        if (gaps.Count != 0) return gaps;

        if (headsKv > heads)
        {
            gaps.Add(new("architecture", "attention.head_count_kv", "greater than attention.head_count"));
            return gaps;
        }
        if (layers > int.MaxValue || embedding > int.MaxValue || feedForward > int.MaxValue
            || heads > int.MaxValue || headsKv > int.MaxValue)
        {
            gaps.Add(new("shape", "dimensions", "a derived dimension exceeds Int32"));
            return gaps;
        }

        // Explicit key/value length wins; otherwise the head dimension divides the embedding.
        int headDim;
        if (config.TryInt("attention.key_length", out long keyLength))
        {
            if (keyLength <= 0 || keyLength > int.MaxValue)
            {
                gaps.Add(new("architecture", "attention.key_length", "non-positive or exceeds Int32"));
                return gaps;
            }
            headDim = (int)keyLength;
        }
        else
        {
            if (embedding % heads != 0)
            {
                gaps.Add(new("architecture", "attention.key_length",
                    $"absent and embedding_length ({embedding}) is not divisible by head_count ({heads})"));
                return gaps;
            }
            headDim = (int)(embedding / heads);
        }

        int vocabulary;
        if (config.TryInt("vocab_size", out long declaredVocab) && declaredVocab > 0)
            vocabulary = declaredVocab > int.MaxValue ? -1 : (int)declaredVocab;
        else if (tensors.TryGetValue("token_embd.weight", out var embeddingTensor) && embeddingTensor.Dimensions.Count == 2)
            vocabulary = embeddingTensor.Dimensions[1];
        else
            vocabulary = -1;
        if (vocabulary <= 0)
        {
            gaps.Add(new("architecture", "vocab_size", "not declared and not derivable from token_embd.weight"));
            return gaps;
        }

        int embeddingInt = (int)embedding;
        int feedForwardInt = (int)feedForward;
        int headsInt = (int)heads;
        int headsKvInt = (int)headsKv;
        int qWidth = Checked(headsInt, headDim, gaps, "attention.key_length");
        int kvWidth = Checked(headsKvInt, headDim, gaps, "attention.key_length");
        if (gaps.Count != 0) return gaps;

        bool hasOutput = tensors.ContainsKey("output.weight");
        bool declaredTied = config.TryBool("tie_word_embeddings", out bool tiedFlag) && tiedFlag;
        bool tied = declaredTied || !hasOutput;
        Require(tensors, gaps, "token_embd.weight", embeddingInt, vocabulary);
        Require(tensors, gaps, "output_norm.weight", embeddingInt);
        if (hasOutput)
            Require(tensors, gaps, "output.weight", embeddingInt, vocabulary);

        for (int layer = 0; layer < layers; layer++)
        {
            string prefix = $"blk.{layer}.";
            Require(tensors, gaps, prefix + "attn_norm.weight", embeddingInt);
            Require(tensors, gaps, prefix + "attn_q.weight", embeddingInt, qWidth);
            Require(tensors, gaps, prefix + "attn_k.weight", embeddingInt, kvWidth);
            Require(tensors, gaps, prefix + "attn_v.weight", embeddingInt, kvWidth);
            Require(tensors, gaps, prefix + "attn_output.weight", qWidth, embeddingInt);
            Require(tensors, gaps, prefix + "ffn_norm.weight", embeddingInt);
            Require(tensors, gaps, prefix + "ffn_gate.weight", embeddingInt, feedForwardInt);
            Require(tensors, gaps, prefix + "ffn_up.weight", embeddingInt, feedForwardInt);
            Require(tensors, gaps, prefix + "ffn_down.weight", feedForwardInt, embeddingInt);
            RequireOptional(tensors, gaps, prefix + "attn_q.bias", qWidth);
            RequireOptional(tensors, gaps, prefix + "attn_k.bias", kvWidth);
            RequireOptional(tensors, gaps, prefix + "attn_v.bias", kvWidth);
            RequireOptional(tensors, gaps, prefix + "attn_q_norm.weight", headDim);
            RequireOptional(tensors, gaps, prefix + "attn_k_norm.weight", headDim);
            if (gaps.Count > 64) break;
        }

        if (gaps.Count == 0)
            shape = new ModelShape((int)layers, embeddingInt, feedForwardInt, headsInt, headsKvInt, headDim,
                vocabulary, tied, epsilon, rope);
        return gaps;
    }

    private static int Checked(int a, int b, List<CapabilityGap> gaps, string detail)
    {
        long product = (long)a * b;
        if (product > int.MaxValue)
        {
            gaps.Add(new("shape", detail, "projection width exceeds Int32"));
            return 0;
        }
        return (int)product;
    }

    private static void Require(
        IReadOnlyDictionary<string, ModelTensor> tensors, List<CapabilityGap> gaps, string name, params int[] dims)
    {
        if (!tensors.TryGetValue(name, out var tensor))
        {
            gaps.Add(new("shape", name, "required tensor is absent"));
            return;
        }
        CheckDimensions(tensor, gaps, name, dims);
    }

    private static void RequireOptional(
        IReadOnlyDictionary<string, ModelTensor> tensors, List<CapabilityGap> gaps, string name, params int[] dims)
    {
        if (tensors.TryGetValue(name, out var tensor))
            CheckDimensions(tensor, gaps, name, dims);
    }

    private static void CheckDimensions(ModelTensor tensor, List<CapabilityGap> gaps, string name, int[] dims)
    {
        if (tensor.Dimensions.Count != dims.Length || !tensor.Dimensions.SequenceEqual(dims))
            gaps.Add(new("shape", name,
                $"declared [{string.Join("x", tensor.Dimensions)}] != expected [{string.Join("x", dims)}]"));
    }
}

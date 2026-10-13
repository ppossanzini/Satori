using Satori.Core.Tokenization;

namespace Satori.Core.Model;

/// <summary>Why generation stopped.</summary>
public enum GenerationStop
{
    /// <summary>The end-of-sequence token was produced.</summary>
    EndOfSequence,

    /// <summary>The requested number of new tokens was reached.</summary>
    TokenLimit,

    /// <summary>The session reached its context limit before the token limit.</summary>
    ContextFull,
}

/// <summary>Generation options. <see cref="Sampling"/> follows <see cref="SamplingOptions"/>: temperature 0 is greedy.</summary>
public sealed record TextGenerationOptions(int MaxNewTokens, SamplingOptions? Sampling = null);

/// <summary>Prompt tokens, produced tokens, decoded text and the stop reason.</summary>
public sealed record TextGenerationResult(
    IReadOnlyList<int> PromptTokens,
    IReadOnlyList<int> GeneratedTokens,
    string Text,
    GenerationStop Stop);

/// <summary>
/// Connects a session to a tokenizer: it encodes the prompt, appends tokens incrementally, samples each next
/// token with the shared <see cref="TokenSampler"/> and stops at the token limit, the context limit or the
/// end-of-sequence token. It is generic over model and tokenizer; no model name appears here.
/// </summary>
public sealed class TextGenerator
{
    private readonly DenseSession _session;
    private readonly ITextTokenizer _tokenizer;
    private readonly int? _endOfSequence;

    public TextGenerator(DenseSession session, ITextTokenizer tokenizer, int? endOfSequenceToken = null)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(tokenizer);
        int vocabulary = session.Model.Shape?.VocabularySize
            ?? throw new InvalidOperationException("The session model has no derived shape.");
        if (endOfSequenceToken is int eos && (uint)eos >= (uint)vocabulary)
            throw new ArgumentOutOfRangeException(nameof(endOfSequenceToken), "End-of-sequence token is outside the vocabulary.");
        _session = session;
        _tokenizer = tokenizer;
        _endOfSequence = endOfSequenceToken;
    }

    /// <summary>Encodes <paramref name="prompt"/> and generates a continuation.</summary>
    public TextGenerationResult Generate(string prompt, TextGenerationOptions options, Action<int>? onToken = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        int[] encoded = [.. _tokenizer.Encode(prompt)];
        return GenerateFromTokens(encoded, options, onToken);
    }

    /// <summary>Generates a continuation from explicit prompt tokens.</summary>
    public TextGenerationResult GenerateFromTokens(ReadOnlySpan<int> promptTokens, TextGenerationOptions options, Action<int>? onToken = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (promptTokens.IsEmpty) throw new ArgumentException("Prompt must contain at least one token.", nameof(promptTokens));
        if (options.MaxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(options), "MaxNewTokens must be non-negative.");
        var sampling = options.Sampling ?? new SamplingOptions();
        sampling.Validate();

        int vocabulary = _session.Model.Shape!.VocabularySize;
        int[] prompt = promptTokens.ToArray();
        _session.Forward(prompt, new float[vocabulary]);

        var sampler = new TokenSampler(vocabulary, sampling);
        var generated = new List<int>(options.MaxNewTokens);
        var stop = GenerationStop.TokenLimit;
        for (int i = 0; i < options.MaxNewTokens; i++)
        {
            if (_session.Position >= _session.ContextLength)
            {
                stop = GenerationStop.ContextFull;
                break;
            }
            int next = sampler.Select(_session.Logits);
            if (_endOfSequence is int eos && next == eos)
            {
                stop = GenerationStop.EndOfSequence;
                break;
            }
            generated.Add(next);
            onToken?.Invoke(next);
            _session.Append(next);
        }
        return new TextGenerationResult(prompt, generated, _tokenizer.Decode(generated), stop);
    }
}

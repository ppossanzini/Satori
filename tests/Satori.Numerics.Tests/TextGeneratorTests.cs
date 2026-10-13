using Satori.Core;
using Satori.Core.Model;
using Satori.Core.Tokenization;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS10: generation wires a session to a tokenizer with greedy sampling by default, stops at the token
/// limit, the context limit or the end-of-sequence token, and matches the shared sampler. Uses synthetic
/// in-memory weights so it runs in the offline suite; the real-model greedy sequence is checked in
/// <see cref="QwenForwardTests"/>.
/// </summary>
public sealed class TextGeneratorTests
{
    private static ModelShape Shape() => new(
        Layers: 2, EmbeddingLength: 32, FeedForwardLength: 48, HeadCount: 4, HeadCountKv: 2, HeadDim: 8,
        VocabularySize: 64, TiedEmbeddings: true, RmsNormEpsilon: 1e-5, RopeBase: 10000);

    private static ModelDefinition Definition(ModelShape shape) =>
        new("test", new ModelConfig("test", new Dictionary<string, object?>()), [], [], shape);

    /// <summary>One id per character code, so encode/decode round-trips deterministically.</summary>
    private sealed class CharTokenizer : ITextTokenizer
    {
        public IReadOnlyList<int> Encode(string text) => text.Select(c => (int)c % 64).ToArray();
        public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true) =>
            string.Concat(ids.Select(id => (char)('a' + id % 26)));
        public bool IsSpecial(int id) => false;
    }

    private static (DenseSession Session, ModelShape Shape) Create(int context)
    {
        var shape = Shape();
        var weights = new SyntheticWeights(shape, bias: true, headNorm: false);
        return (new DenseSession(Definition(shape), weights, context), shape);
    }

    [Fact]
    public void GreedyGenerationEqualsTheReferenceLoopAndStopsAtTheTokenLimit()
    {
        var (session, shape) = Create(context: 64);
        var generator = new TextGenerator(session, new CharTokenizer());
        var result = generator.GenerateFromTokens([1, 2, 3], new TextGenerationOptions(5));

        Assert.Equal(GenerationStop.TokenLimit, result.Stop);
        Assert.Equal([1, 2, 3], result.PromptTokens);
        Assert.Equal(5, result.GeneratedTokens.Count);
        Assert.Equal(8, session.Position);

        // Recompute the same greedy sequence with a fresh session and the shared sampler.
        var (reference, _) = Create(context: 64);
        reference.Forward([1, 2, 3], new float[shape.VocabularySize]);
        var expected = new List<int>();
        for (int i = 0; i < 5; i++)
        {
            int next = GreedyTokenSelector.Select(reference.Logits);
            expected.Add(next);
            reference.Append(next);
        }
        Assert.Equal(expected, result.GeneratedTokens);
        Assert.Equal(0.0, new SamplingOptions().Temperature);
    }

    [Fact]
    public void EndOfSequenceStopsImmediatelyAfterTheFirstToken()
    {
        var (session, _) = Create(context: 64);
        var first = new TextGenerator(session, new CharTokenizer()).GenerateFromTokens([4], new TextGenerationOptions(4));
        int eos = first.GeneratedTokens[0];

        var (session2, _) = Create(context: 64);
        var result = new TextGenerator(session2, new CharTokenizer(), eos)
            .GenerateFromTokens([4], new TextGenerationOptions(4));
        Assert.Equal(GenerationStop.EndOfSequence, result.Stop);
        Assert.Empty(result.GeneratedTokens);
        Assert.Equal(1, session2.Position);
    }

    [Fact]
    public void ContextFullStopsBeforeTheTokenLimit()
    {
        var (session, _) = Create(context: 4);
        var result = new TextGenerator(session, new CharTokenizer())
            .GenerateFromTokens([1, 2], new TextGenerationOptions(10));
        Assert.Equal(GenerationStop.ContextFull, result.Stop);
        Assert.Equal(2, result.GeneratedTokens.Count);
        Assert.Equal(4, session.Position);
    }

    [Fact]
    public void SeededSamplingMatchesTheSharedSampler()
    {
        var options = new TextGenerationOptions(6, new SamplingOptions(Temperature: 0.8, TopK: 10, Seed: 99));
        var (session, shape) = Create(context: 32);
        var result = new TextGenerator(session, new CharTokenizer()).GenerateFromTokens([7, 8], options);

        var (reference, _) = Create(context: 32);
        reference.Forward([7, 8], new float[shape.VocabularySize]);
        var sampler = new TokenSampler(shape.VocabularySize, options.Sampling);
        var expected = new List<int>();
        for (int i = 0; i < 6; i++)
        {
            int next = sampler.Select(reference.Logits);
            expected.Add(next);
            reference.Append(next);
        }
        Assert.Equal(expected, result.GeneratedTokens);
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        var (session, _) = Create(context: 16);
        var generator = new TextGenerator(session, new CharTokenizer());
        Assert.Throws<ArgumentException>(() => generator.GenerateFromTokens([], new TextGenerationOptions(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => generator.GenerateFromTokens([1], new TextGenerationOptions(-1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => generator.GenerateFromTokens([64], new TextGenerationOptions(1)));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new TextGenerator(session, new CharTokenizer(), endOfSequenceToken: 64 * 100));
    }
}

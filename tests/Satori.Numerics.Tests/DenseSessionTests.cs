using Satori.Core.Model;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS09: the incremental session must agree with full recomputation, honour its context limit and lifetime,
/// and size its state like the plan. Uses synthetic in-memory weights so it runs in the offline suite; the
/// real-model parity is covered by <see cref="QwenForwardTests"/>.
/// </summary>
public sealed class DenseSessionTests
{
    private static ModelShape Shape(bool tied) => new(
        Layers: 3, EmbeddingLength: 32, FeedForwardLength: 64, HeadCount: 4, HeadCountKv: 2, HeadDim: 8,
        VocabularySize: 32, TiedEmbeddings: tied, RmsNormEpsilon: 1e-5, RopeBase: 10000);

    private static ModelDefinition Definition(ModelShape shape) =>
        new("test", new ModelConfig("test", new Dictionary<string, object?>()), [], [], shape);

    public static TheoryData<bool, bool, bool> Variants => new()
    {
        { false, true, false },  // qwen2-like: QKV bias, no Q/K norm
        { true, false, true },   // qwen3-like: Q/K norm, tied embeddings
    };

    [Theory]
    [MemberData(nameof(Variants))]
    public void IncrementalLogitsMatchFullRecomputation(bool tied, bool bias, bool headNorm)
    {
        var shape = Shape(tied);
        var weights = new SyntheticWeights(shape, bias, headNorm);
        var definition = Definition(shape);
        var forward = new DenseForward(shape, weights);
        int[] prompt = [3, 7, 11, 5, 19];
        int next = 23;

        float[] full = forward.Logits(prompt);
        using var session = new DenseSession(definition, weights, contextLength: 16);
        float[] sessionLogits = new float[shape.VocabularySize];
        session.Forward(prompt, sessionLogits);
        Assert.Equal(prompt.Length, session.Position);
        AssertClose(full, sessionLogits, 1e-6);

        // Appending one token must equal recomputing the whole prefix plus that token.
        session.Append(next);
        float[] extended = forward.Logits([.. prompt, next]);
        Assert.Equal(prompt.Length + 1, session.Position);
        AssertClose(extended, session.Logits.ToArray(), 1e-6);
    }

    [Fact]
    public void ResetReturnsToTheInitialState()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: true, headNorm: false);
        using var session = new DenseSession(Definition(shape), weights, contextLength: 8);
        session.Append(1);
        session.Append(2);
        float[] first = session.Logits.ToArray();

        session.Reset();
        Assert.Equal(0, session.Position);
        session.Append(1);
        session.Append(2);
        AssertClose(first, session.Logits.ToArray(), 0);
    }

    [Fact]
    public void StateBytesMatchThePlanAndContextLimitIsEnforced()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: true, headNorm: false);
        var definition = Definition(shape);
        using var session = new DenseSession(definition, weights, contextLength: 6);
        Assert.Equal(ModelPlanner.Plan(shape, 0, 6, 0).StateBytes, session.StateBytes);
        Assert.Equal(6, session.ContextLength);

        float[] logits = new float[shape.VocabularySize];
        session.Forward([1, 2, 3, 4], logits);
        session.Forward([5, 6], logits);
        Assert.Throws<InvalidOperationException>(() => session.Append(7));
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Forward([7, 8], logits));
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Append(shape.VocabularySize));
        Assert.Equal(6, session.Position);
    }

    /// <summary>
    /// Two hundred positions cross the prefill chunk boundary at one thread (64 positions per chunk) and fit
    /// one chunk at four threads (256), so both the chunked and the single-chunk path are covered.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void BatchedPrefillMatchesSequentialAppend(int threads)
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: true, headNorm: false);
        var definition = Definition(shape);
        int[] prompt = [.. Enumerable.Range(0, 200).Select(i => (i * 7 + 3) % shape.VocabularySize)];

        using var batched = new DenseSession(definition, weights, contextLength: 256, options: new InferenceOptions(threads));
        float[] batchedLogits = new float[shape.VocabularySize];
        batched.Forward(prompt, batchedLogits);

        using var sequential = new DenseSession(definition, weights, contextLength: 256, options: new InferenceOptions(1));
        foreach (int token in prompt) sequential.Append(token);

        Assert.Equal(prompt.Length, batched.Position);
        Assert.Equal(sequential.Position, batched.Position);
        Assert.Equal(sequential.Logits.ToArray(), batchedLogits);
    }

    /// <summary>The decoded continuation continues from the prefilled state, not from a fresh one.</summary>
    [Fact]
    public void AppendAfterABatchedPrefillMatchesFullRecomputation()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: false, headNorm: true);
        var definition = Definition(shape);
        int[] prompt = [4, 9, 14, 21, 2, 8];
        int next = 17;

        using var session = new DenseSession(definition, weights, contextLength: 32, options: new InferenceOptions(3));
        session.Forward(prompt, new float[shape.VocabularySize]);
        session.Append(next);

        float[] full = new DenseForward(shape, weights).Logits([.. prompt, next]);
        AssertClose(full, session.Logits.ToArray(), 1e-6);
    }

    [Fact]
    public void ThreadCountIsValidatedAndBoundedByTheProtocolBudget()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: false, headNorm: false);
        var definition = Definition(shape);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DenseSession(definition, weights, 4, options: new InferenceOptions(0)));
        Assert.True(InferenceOptions.Default.Threads >= 1);
        Assert.True(InferenceOptions.Default.Threads <= InferenceOptions.BudgetThreads);
    }

    [Fact]
    public void DisposeIsIdempotentAndRejectsFurtherUse()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: false, headNorm: false);
        var session = new DenseSession(Definition(shape), weights, contextLength: 4);
        session.Dispose();
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.Append(1));
        Assert.Throws<ObjectDisposedException>(() => session.Forward([1], new float[shape.VocabularySize]));
    }

    [Fact]
    public void NonExecutableDefinitionIsRejected()
    {
        var shape = Shape(false);
        var weights = new SyntheticWeights(shape, bias: false, headNorm: false);
        var withGap = new ModelDefinition("test", new ModelConfig("test", new Dictionary<string, object?>()),
            [], [new CapabilityGap("tensor-format", "Q6_K(14)", "unsupported")], shape);
        Assert.Throws<NotSupportedException>(() => new DenseSession(withGap, weights, 4));

        var noShape = new ModelDefinition("test", new ModelConfig("test", new Dictionary<string, object?>()), [], [], null);
        Assert.Throws<InvalidOperationException>(() => new DenseSession(noShape, weights, 4));
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance + tolerance * Math.Abs(expected[i]),
                $"[{i}] {actual[i]} vs {expected[i]}");
    }
}

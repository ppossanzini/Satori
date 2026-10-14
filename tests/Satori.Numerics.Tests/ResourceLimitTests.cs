using Satori.Core.Model;

namespace Satori.Numerics.Tests;

/// <summary>
/// DW06.8: the session must refuse work it cannot fit before it allocates, and must honour a request to stop
/// within a bounded time without committing a half-written position.
/// </summary>
/// <remarks>
/// What is asserted here is structural and deterministic: that the limit is applied before allocation, that a
/// stop request raised at any point of a layer is observed, and that an interrupted run leaves the session at
/// a committed boundary. The latency itself is a duration and belongs to the harness, not the suite — see rule
/// 6 of docs/measurement-method.md, written after a duration assertion in this suite failed on a descheduling
/// outlier. The console reports the latency it measures on a real model.
/// </remarks>
public sealed class ResourceLimitTests
{
    private static ModelShape Shape() => new(
        Layers: 3, EmbeddingLength: 32, FeedForwardLength: 64, HeadCount: 4, HeadCountKv: 2, HeadDim: 8,
        VocabularySize: 32, TiedEmbeddings: true, RmsNormEpsilon: 1e-5, RopeBase: 10000);

    private static ModelDefinition Definition(ModelShape shape) =>
        new("test", new ModelConfig("test", new Dictionary<string, object?>()), [], [], shape);

    private static SyntheticWeights Weights(ModelShape shape) => new(shape, bias: true, headNorm: false);

    /// <summary>
    /// A budget below the plan is refused, and refused before the key/value state is allocated: the message
    /// names the three accounts so an operator can see which one to reduce.
    /// </summary>
    [Fact]
    public void ABudgetBelowThePlanIsRefusedAndSaysWhy()
    {
        var shape = Shape();
        var weights = Weights(shape);
        var definition = Definition(shape);
        const int context = 64;

        ModelPlan plan = definition.Plan(context, budgetBytes: 0);
        long tooSmall = plan.TotalBytes - 1;
        var exception = Assert.Throws<InvalidOperationException>(() =>
            new DenseSession(definition, weights, context, options: new InferenceOptions(1) { BudgetBytes = tooSmall }));
        Assert.Contains("budget is", exception.Message, StringComparison.Ordinal);
        Assert.Contains("weights", exception.Message, StringComparison.Ordinal);
        Assert.Contains("state", exception.Message, StringComparison.Ordinal);
        Assert.Contains("workspace", exception.Message, StringComparison.Ordinal);
    }

    /// <summary>A budget that exactly covers the plan is honoured, so the boundary is inclusive and the
    /// refusal above is off by one and not by a policy margin.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void APlanThatFitsIsAccepted(int slack)
    {
        var shape = Shape();
        var weights = Weights(shape);
        var definition = Definition(shape);
        const int context = 64;
        ModelPlan plan = definition.Plan(context, budgetBytes: 0);

        using var session = new DenseSession(definition, weights, context,
            options: new InferenceOptions(1) { BudgetBytes = plan.TotalBytes + slack });
        Assert.Equal(plan.TotalBytes + slack, session.BudgetBytes);
        session.Append(1);
        Assert.Equal(1, session.Position);
    }

    /// <summary>Zero states no limit, which is what the host gets when it says nothing.</summary>
    [Fact]
    public void NoBudgetMeansNoLimit()
    {
        var shape = Shape();
        var weights = Weights(shape);
        using var session = new DenseSession(Definition(shape), weights, contextLength: 64);
        Assert.Equal(0, session.BudgetBytes);
    }

    /// <summary>A negative budget is a caller error, not a licence to allocate without limit.</summary>
    [Fact]
    public void ANegativeBudgetIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceOptions(1) { BudgetBytes = -1 }.Validated());
        Assert.Equal(0, InferenceOptions.Default.BudgetBytes);
    }

    /// <summary>A token already cancelled aborts before the first weight byte is read.</summary>
    [Fact]
    public void AnAlreadyCancelledRequestAbortsImmediately()
    {
        var shape = Shape();
        var weights = Weights(shape);
        using var session = new DenseSession(Definition(shape), weights, contextLength: 64);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        float[] logits = new float[shape.VocabularySize];
        Assert.Throws<OperationCanceledException>(() => session.Forward([1, 2, 3, 4], logits, cancellation.Token));
        Assert.Throws<OperationCanceledException>(() => session.Append(1, cancellation.Token));
        Assert.Equal(0, session.Position);
    }

    /// <summary>
    /// The bound, tested as a property rather than as a duration: a stop request raised at any point the
    /// session reports progress from must be observed. If some stretch of a layer ran with no checkpoint, a
    /// cancellation raised there would run to the end of that stretch, and this is what would fail.
    /// </summary>
    /// <remarks>
    /// The stages come from the session's own observer, so the test cannot go stale when the layer gains a
    /// phase: a new stage is exercised automatically and a phase with no checkpoint after it is caught.
    /// </remarks>
    [Fact]
    public void AStopRequestRaisedAtAnyReportedStageIsObserved()
    {
        var shape = Shape();
        var weights = Weights(shape);
        var definition = Definition(shape);
        const int context = 16;
        int[] prompt = [1, 2, 3, 4, 5, 6, 7, 8];

        // First pass: discover the stages the forward pass reports, in order, and how often each occurs.
        var seen = new List<string>();
        using (var probe = new DenseSession(definition, weights, context,
            observe: (_, _, stage, _) => seen.Add(stage)))
        {
            probe.Forward(prompt, new float[shape.VocabularySize]);
        }
        Assert.NotEmpty(seen);

        // Second pass: raise the request on the n-th report of each stage and require it to be observed.
        var stages = seen.Distinct().ToArray();
        foreach (string stage in stages)
        {
            foreach (int occurrence in new[] { 1, seen.Count(stage2 => stage2 == stage) })
            {
                using var cancellation = new CancellationTokenSource();
                int count = 0;
                using var session = new DenseSession(definition, weights, context, observe: (_, _, observed, _) =>
                {
                    if (observed == stage && ++count == occurrence) cancellation.Cancel();
                });
                float[] logits = new float[shape.VocabularySize];
                Assert.Throws<OperationCanceledException>(() => session.Forward(prompt, logits, cancellation.Token));
            }
        }
    }

    /// <summary>
    /// An interrupted prefill commits nothing. The chunk boundary is the commit point, so a session stopped
    /// mid-prompt is at its previous position rather than holding a prefix that looks committed.
    /// </summary>
    [Fact]
    public void AnInterruptedPrefillLeavesNoPartialPosition()
    {
        var shape = Shape();
        var weights = Weights(shape);
        var definition = Definition(shape);
        const int context = 512;
        int[] prompt = [.. Enumerable.Range(1, 300).Select(value => value % shape.VocabularySize)];

        using var cancellation = new CancellationTokenSource();
        using var session = new DenseSession(definition, weights, context, observe: (layer, position, stage, _) =>
        {
            if (stage == "out" && position > 200) cancellation.Cancel();
        });
        float[] logits = new float[shape.VocabularySize];
        Assert.Throws<OperationCanceledException>(() => session.Forward(prompt, logits, cancellation.Token));
        Assert.Equal(0, session.Position);
    }

    /// <summary>An interrupted session is still usable: cancellation is a stop, not a broken state.</summary>
    [Fact]
    public void AnInterruptedSessionRemainsUsable()
    {
        var shape = Shape();
        var weights = Weights(shape);
        var definition = Definition(shape);
        using var cancellation = new CancellationTokenSource();
        using var session = new DenseSession(definition, weights, contextLength: 64, observe: (_, _, stage, _) =>
        {
            if (stage == "out") cancellation.Cancel();
        });
        float[] logits = new float[shape.VocabularySize];
        Assert.Throws<OperationCanceledException>(() => session.Forward([1, 2, 3, 4], logits, cancellation.Token));

        using var clean = new CancellationTokenSource();
        session.Forward([1, 2, 3, 4], logits, clean.Token);
        Assert.Equal(4, session.Position);
        Assert.Contains(logits, value => value != 0);
    }

    /// <summary>
    /// The generator stops between tokens as well as inside them, so a request raised after a token cannot be
    /// followed by another token of work.
    /// </summary>
    [Fact]
    public void TheGeneratorStopsBetweenTokens()
    {
        var shape = Shape();
        var weights = Weights(shape);
        using var session = new DenseSession(Definition(shape), weights, contextLength: 64);
        var generator = new TextGenerator(session, new CharTokenizer());
        using var cancellation = new CancellationTokenSource();
        int produced = 0;
        Assert.Throws<OperationCanceledException>(() => generator.GenerateFromTokens([1, 2, 3, 4],
            new TextGenerationOptions(64), _ => { if (++produced == 3) cancellation.Cancel(); }, cancellation.Token));
        Assert.Equal(3, produced);
        Assert.Equal(4 + 2, session.Position);
    }

    private sealed class CharTokenizer : Satori.Core.Tokenization.ITextTokenizer
    {
        public IReadOnlyList<int> Encode(string text) => text.Select(c => (int)c % 32).ToArray();
        public string Decode(IReadOnlyList<int> ids, bool skipSpecialTokens = true) => string.Concat(ids.Select(id => 'a'));
        public bool IsSpecial(int id) => false;
    }
}

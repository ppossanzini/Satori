using Satori.Core.Model;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

/// <summary>
/// Guards the thread budget's contract: the host states a count or takes the default, and only a count that
/// cannot be served is refused.
/// </summary>
/// <remarks>
/// A physical-core clamp was implemented and withdrawn on measurement — see
/// <see cref="InferenceOptions.Default"/>. These tests pin the contract that survived, not the clamp.
/// </remarks>
public sealed class InferenceOptionsTests
{
    private readonly ITestOutputHelper _output;

    public InferenceOptionsTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// An explicit count is the host's decision and is not clamped to the topology: only a count that cannot
    /// be served is rejected. A host that asks for more workers than the machine has may know something the
    /// runtime does not, so the runtime must not silently overrule it.
    /// </summary>
    [Fact]
    public void AnExplicitCountIsAcceptedAndOnlyAnUnservableOneIsRejected()
    {
        int above = Environment.ProcessorCount + 4;
        Assert.Equal(above, new InferenceOptions(above).Validated().Threads);
        Assert.Equal(1, new InferenceOptions(1).Validated().Threads);
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceOptions(0).Validated());
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceOptions(-1).Validated());
    }

    /// <summary>
    /// The default is the budget where the machine can serve it, and never zero or negative: a session that
    /// cannot run a projection cannot run at all.
    /// </summary>
    [Fact]
    public void TheDefaultIsTheBudgetWhereTheMachineCanServeIt()
    {
        int threads = InferenceOptions.Default.Threads;
        _output.WriteLine($"logical={Environment.ProcessorCount} budget={threads}");
        Assert.Equal(Math.Min(InferenceOptions.BudgetThreads, Environment.ProcessorCount), threads);
        Assert.InRange(threads, 1, Environment.ProcessorCount);
    }

    /// <summary>
    /// The default is already valid, so a session built without options never has to defend itself against its
    /// own fallback.
    /// </summary>
    [Fact]
    public void TheDefaultIsAlreadyValid()
    {
        InferenceOptions validated = InferenceOptions.Default.Validated();
        Assert.Equal(InferenceOptions.Default.Threads, validated.Threads);
    }
}

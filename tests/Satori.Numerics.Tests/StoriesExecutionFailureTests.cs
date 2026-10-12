using Satori.Core;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class StoriesExecutionFailureTests
{
    [Fact]
    public void FailedAppendClearsCacheBlocksContinuationAndReleasesOwnedMatrices()
    {
        var factory=new FailingMatrixFactory {FailExecution=false,CallsBeforeFailure=53};
        using var model=new StoriesModel(Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf"),factory);
        using var session=model.CreateSession();
        float[] logits=Enumerable.Repeat(123f,32000).ToArray();
        Assert.True(session.TryAppend(1,logits));
        Assert.Equal(1,session.Count);
        Array.Fill(logits,123f);
        factory.FailExecution=true;
        Assert.Throws<IOException>(()=>session.TryAppend(9038,logits));
        Assert.Equal(0,session.Count);
        Assert.All(logits,v=>Assert.Equal(123f,v));
        Assert.Throws<InvalidOperationException>(()=>session.TryAppend(1,logits));
        session.Reset();
        factory.FailExecution=false;
        Assert.True(session.TryAppend(1,logits));
        using var cpu=new StoriesModel(Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf"));
        float[] expected=new float[32000];cpu.Forward([1],expected);
        Assert.Equal(expected,logits);
        Assert.Equal(1,session.Count);
        model.Dispose();model.Dispose();
        Assert.Equal(43,factory.Disposed);
    }

    [Fact]
    public void FailedResidentConstructionReleasesPreviouslyCreatedMatrices()
    {
        var factory=new FailingMatrixFactory(7);
        Assert.Throws<IOException>(()=>new StoriesModel(Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf"),factory));
        Assert.Equal(7,factory.Disposed);
    }
}

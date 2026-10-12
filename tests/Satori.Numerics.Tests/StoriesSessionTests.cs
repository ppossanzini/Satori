using System.Text.Json;
using Satori.Formats.Gguf;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

[Trait("Category", "ModelIntegration")]
public sealed class StoriesSessionTests(ITestOutputHelper output)
{
    private static string ModelPath => Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf");

    [Theory]
    [InlineData("empty",false)]
    [InlineData("story",false)]
    [InlineData("italian",false)]
    [InlineData("empty",true)]
    [InlineData("story",true)]
    [InlineData("italian",true)]
    public void EachIncrementalLogitMatchesFullPrefixExactly(string id, bool scalar)
    {
        using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/numpy/reference.json")));
        var item = json.RootElement.GetProperty("forward_cases").EnumerateArray().Single(v=>v.GetProperty("id").GetString()==id);
        int[] tokens = item.GetProperty("tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        using var model = new StoriesModel(ModelPath);
        using var session = model.CreateSession(scalar);
        float[] incremental = new float[32000], reference = new float[32000];
        Assert.Equal(0,session.Count);
        for (int i = 0; i < tokens.Length; i++)
        {
            Assert.True(session.TryAppend(tokens[i],incremental));
            model.Forward(tokens.AsSpan(0,i+1),reference,scalar);
            Assert.Equal(reference,incremental);
            Assert.Equal(i+1,session.Count);
        }
        session.Reset();
        Assert.Equal(0,session.Count);
        Assert.True(session.TryAppend(1,incremental));
        model.Forward([1],reference,scalar);
        Assert.Equal(reference,incremental);
    }

    [Fact]
    public void ContextLimitResetAndAllocationBudgetHoldAtEveryBoundary()
    {
        using var model = new StoriesModel(ModelPath);
        using (var warm = model.CreateSession()) { warm.TryAppend(1,new float[32000]); warm.Reset(); }
        long before = GC.GetAllocatedBytesForCurrentThread();
        using var session = model.CreateSession();
        long construction = GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.True(construction<=2*1024*1024,$"Construction allocated {construction} bytes.");
        Assert.Equal(1769472,StoriesSession.CacheByteLength);
        int[] tokens = Enumerable.Range(0,128).Select(i=>i==0 ? 1 : i%2==0 ? 9038 : 2501).ToArray();
        float[] incremental = new float[32000], reference = new float[32000];
        for (int i = 0; i < tokens.Length; i++)
        {
            before = GC.GetAllocatedBytesForCurrentThread();
            bool appended = session.TryAppend(tokens[i],incremental);
            long allocated = GC.GetAllocatedBytesForCurrentThread()-before;
            Assert.True(appended);
            Assert.Equal(0,allocated);
            if (i is 0 or 1 or 31 or 63 or 127)
            {
                model.Forward(tokens.AsSpan(0,i+1),reference);
                Assert.Equal(reference,incremental);
            }
        }
        Assert.Equal(128,session.Count);
        float[] saved = incremental.ToArray();
        Assert.False(session.TryAppend(9038,incremental));
        Assert.Equal(saved,incremental);
        Assert.Equal(128,session.Count);
        before = GC.GetAllocatedBytesForCurrentThread();
        session.Reset();
        long resetAllocated = GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.Equal(0,resetAllocated);
        Assert.True(session.TryAppend(1,incremental));
        Assert.True(session.TryAppend(29892,incremental));
        model.Forward([1,29892],reference);
        Assert.Equal(reference,incremental);
        output.WriteLine($"KV payload: {StoriesSession.CacheByteLength}; construction: {construction} bytes; append/reset: 0 allocated bytes.");
    }

    [Fact]
    public void InvalidCallsDoNotAdvanceAndLifetimeRemainsExplicit()
    {
        using var model = new StoriesModel(ModelPath);
        using var session = model.CreateSession();
        float[] logits = Enumerable.Repeat(123f,32000).ToArray();
        Assert.Throws<ArgumentOutOfRangeException>(()=>session.TryAppend(-1,logits));
        Assert.Throws<ArgumentOutOfRangeException>(()=>session.TryAppend(32000,logits));
        Assert.Throws<ArgumentException>(()=>session.TryAppend(1,new float[31999]));
        Assert.Equal(0,session.Count);
        Assert.All(logits,value=>Assert.Equal(123f,value));
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(()=>session.TryAppend(1,logits));
        Assert.Equal(0,session.Count);
        Assert.All(logits,value=>Assert.Equal(123f,value));
        Assert.Throws<ObjectDisposedException>(()=>model.CreateSession());
        session.Reset();
        session.Dispose();
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(()=>session.TryAppend(1,logits));
        Assert.Throws<ObjectDisposedException>(()=>session.Reset());
        Assert.Throws<ObjectDisposedException>(()=>session.Count);
    }

    [Fact]
    public async Task ConcurrentSessionsKeepSeparatePrefixesAndDisposalDoesNotCloseModel()
    {
        using var model = new StoriesModel(ModelPath);
        using var left = model.CreateSession();
        using var right = model.CreateSession();
        float[] a = new float[32000], b = new float[32000], expected = new float[32000];
        await Task.WhenAll(Task.Run(()=>{ left.TryAppend(1,a); left.TryAppend(9038,a); }),
            Task.Run(()=>{ right.TryAppend(1,b); right.TryAppend(29892,b); right.TryAppend(450,b); }));
        model.Forward([1,9038],expected); Assert.Equal(expected,a);
        model.Forward([1,29892,450],expected); Assert.Equal(expected,b);
        Assert.Equal(2,left.Count); Assert.Equal(3,right.Count);
        left.Dispose();
        Assert.True(right.TryAppend(9038,b));
        model.Forward([1,29892,450,9038],expected); Assert.Equal(expected,b);
    }
}

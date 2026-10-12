using System.Text.Json;
using Satori.Backends.Vulkan;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanAttentionTests
{
    [Fact]
    public void IndependentBoundaryReferencesIsolationResetLimitAndLifetimeHold()
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        Assert.True(VulkanContext.TryCreate(new(Validation:true),out var created,out var availability),availability.ToString());
        using var context=created!;Assert.False(context.Info.Software);
        using var first=new VulkanAttentionSession(context);using var second=new VulkanAttentionSession(context);
        Assert.Equal(2*128*288*4,first.CacheBytes);
        using var fixture=JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"../../../../../fixtures/operators/attention-boundaries-v1.json")));
        var expected=fixture.RootElement.GetProperty("cases").EnumerateArray().ToDictionary(c=>c.GetProperty("count").GetInt32(),c=>c.GetProperty("expected").EnumerateArray().Select(v=>v.GetDouble()).ToArray());
        float[] q=Enumerable.Range(0,288).Select(i=>(i%17-8)*.125f).ToArray(),actual=new float[288],other=new float[288];
        for(int token=0;token<128;token++)
        {
            float[] k=Enumerable.Range(0,288).Select(i=>((token*13+i*7)%31-15)*.0625f).ToArray();
            float[] v=Enumerable.Range(0,288).Select(i=>((token*5+i*11)%37-18)*.125f).ToArray();
            Assert.True(first.TryAppend(q,k,v,actual));
            Assert.True(second.TryAppend(q,k,v.Select(x=>-x).ToArray(),other));
            if(expected.TryGetValue(token+1,out var reference))for(int i=0;i<288;i++)
            {
                Assert.True(Math.Abs(actual[i]-reference[i])<=2e-6+2e-6*Math.Abs(reference[i]));
                Assert.True(Math.Abs(other[i]+reference[i])<=2e-6+2e-6*Math.Abs(reference[i]));
            }
        }
        var saved=actual.ToArray();Assert.False(first.TryAppend(q,q,q,actual));Assert.Equal(saved,actual);Assert.Equal(128,first.Count);
        first.Reset();Assert.Equal(0,first.Count);Assert.True(first.TryAppend(q,q,q,actual));Assert.Equal(q,actual);Assert.Equal(128,second.Count);
        context.Dispose();Assert.Throws<ObjectDisposedException>(()=>first.TryAppend(q,q,q,actual));Assert.Equal(q,actual);Assert.Equal(0,context.ValidationErrorCount);
    }
}

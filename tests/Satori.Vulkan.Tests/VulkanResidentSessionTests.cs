using Satori.Backends.Vulkan;
using Satori.Formats.Gguf;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanResidentSessionTests
{
    [Fact]
    public async Task IndependentSessionsHaveNoHotAllocationAndSerializeOwnerDisposal()
    {
        string path=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"../../../../../.temp/gguf-corpus/stories15M-q8_0.gguf"));
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        using var runtime=new VulkanStoriesRuntime(path,new(StoriesBackendMode.VulkanRequired,new(Validation:true),Path:StoriesVulkanPath.Resident));
        using var first=runtime.Model.CreateSession();using var second=runtime.Model.CreateSession();using var cpu=new StoriesModel(path);using var reference=cpu.CreateSession();
        float[] actual=new float[32000],expected=new float[32000];
        for(int i=0;i<30;i++)first.TryAppend(i==0?1:9038,actual);
        first.Reset();
        long before=GC.GetAllocatedBytesForCurrentThread();
        for(int i=0;i<8;i++)first.TryAppend(i==0?1:9038,actual);
        Assert.Equal(0,GC.GetAllocatedBytesForCurrentThread()-before);
        second.TryAppend(1,actual);reference.TryAppend(1,expected);Compare(actual,expected);
        second.TryAppend(315,actual);reference.TryAppend(315,expected);Compare(actual,expected);Assert.Equal(8,first.Count);Assert.Equal(2,second.Count);
        Assert.Equal(2,runtime.ResidentStatistics!.Sessions);Assert.Equal(2UL*(1909376+129152),runtime.ResidentStatistics.SessionAllocationBytes);
        using var start=new ManualResetEventSlim();
        var work=Task.Run(()=>{start.Wait();try {first.TryAppend(9038,actual);}catch(ObjectDisposedException){}});
        var dispose=Task.Run(()=>{start.Wait();runtime.Dispose();});start.Set();await Task.WhenAll(work,dispose);
        var saved=actual.ToArray();var error=Record.Exception(()=>first.TryAppend(1,actual));Assert.True(error is ObjectDisposedException or InvalidOperationException);Assert.Equal(saved,actual);Assert.Equal(0,first.Count);
        Assert.Throws<InvalidOperationException>(()=>first.TryAppend(1,actual));
        Assert.Equal(0,runtime.ValidationErrorCount);
    }
    private static void Compare(float[] actual,float[] expected)
    {
        double squared=0;for(int i=0;i<actual.Length;i++){double error=Math.Abs(actual[i]-expected[i]);Assert.True(float.IsFinite(actual[i])&&error<=5e-4+5e-5*Math.Abs(expected[i]));squared+=error*error;}
        Assert.True(Math.Sqrt(squared/actual.Length)<=1e-4);
    }
}

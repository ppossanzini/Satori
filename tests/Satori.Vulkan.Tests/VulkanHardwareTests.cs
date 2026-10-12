using Satori.Backends.Vulkan;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanHardwareTests
{
    private static VulkanContext Create()
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        Assert.True(VulkanContext.TryCreate(new(Validation:true),out var context,out var availability),availability.ToString());
        Assert.False(context!.Info.Software);
        return context;
    }

    [Fact]
    public void HardwareTransformCoversTailsGrowthReuseAndRejectionWithoutOutputCommit()
    {
        using var context=Create();
        foreach(int count in new[] {0,1,63,64,65,4097,64,1})
        for(int repeat=0;repeat<5;repeat++)
        {
            float[] input=Enumerable.Range(0,count).Select(i=>(i%137-68)*0.25f+repeat).ToArray();
            float[] output=Enumerable.Repeat(-999f,count).ToArray();
            context.Transform(input,output);
            Assert.Equal(input.Select(v=>2*v+1),output);
        }
        float[] sentinel=[17,17];
        Assert.Throws<ArgumentException>(()=>context.Transform([1],sentinel));
        Assert.Equal(new float[] {17,17},sentinel);
        Assert.Throws<ArgumentException>(()=>context.Transform(sentinel,sentinel));
        Assert.Equal(new float[] {17,17},sentinel);
        Assert.True(context.AllocatedBytes>=4*4097*sizeof(float));
        context.Dispose(); context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
        Assert.False(context.IsFaulted);
        Assert.Throws<ObjectDisposedException>(()=>context.Transform([],[]));
    }

    [Fact]
    public void WarmDispatchReusesOwnedScratchWithoutManagedAllocations()
    {
        using var context=Create();
        float[] input=new float[65],output=new float[65];
        for(int i=0;i<100;i++) context.Transform(input,output);
        ulong scratch=context.AllocatedBytes;
        long before=GC.GetAllocatedBytesForCurrentThread();
        for(int i=0;i<10;i++) context.Transform(input,output);
        long allocated=GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.Equal(0,allocated);
        Assert.Equal(scratch,context.AllocatedBytes);
        context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public async Task ConcurrentDispatchAndDisposeSerializeWithoutInvalidHandles()
    {
        var context=Create();
        await Task.WhenAll(Enumerable.Range(0,4).Select(worker=>Task.Run(()=>
        {
            for(int run=0;run<10;run++)
            {
                float[] input=Enumerable.Repeat((float)worker+run,65).ToArray();
                float[] output=new float[65];
                context.Transform(input,output);
                Assert.All(output,v=>Assert.Equal(2*input[0]+1,v));
            }
        })));
        using var start=new ManualResetEventSlim();
        var workers=Enumerable.Range(0,4).Select(worker=>Task.Run(()=>
        {
            start.Wait();
            for(int run=0;run<20;run++)
            {
                float[] input=Enumerable.Repeat((float)worker+run,65).ToArray();
                float[] output=new float[65];
                try { context.Transform(input,output); Assert.All(output,v=>Assert.Equal(2*input[0]+1,v)); }
                catch(ObjectDisposedException) { return; }
            }
        })).ToArray();
        var disposal=Task.Run(()=> {start.Wait();context.Dispose();});
        start.Set();
        await Task.WhenAll(workers.Append(disposal));
        context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public void RepeatedConstructionAndUnsupportedSelectionCleanUpPartialInstances()
    {
        for(int i=0;i<4;i++)
        {
            Assert.False(VulkanContext.TryCreate(new(int.MaxValue,true),out var absent,out var availability));
            Assert.Null(absent); Assert.Equal(VulkanAvailabilityReason.UnsupportedDevice,availability.Reason);
            using var context=Create();
            float[] output=new float[1];
            context.Transform([i],output);
            Assert.Equal(2*i+1,output[0]);
            context.Dispose();
            Assert.Equal(0,context.ValidationErrorCount);
        }
    }
}

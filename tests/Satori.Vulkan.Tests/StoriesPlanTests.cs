using Satori.Backends.Vulkan;
using System.Text.Json;

namespace Satori.Vulkan.Tests;

public sealed class StoriesPlanTests
{
    [Fact]
    public void MedianRejectsOutlierAndRequiresMeasuredAdvantage()
    {
        var cpu=new double[] {10,10,10,10,1000};
        Assert.Equal("CPU",new StoriesExecutionPlan("key",cpu,[9.6,9.6,9.6,9.6,0.1]).SelectedBackend);
        Assert.Equal("Vulkan",new StoriesExecutionPlan("key",cpu,[9,9,9,9,10000]).SelectedBackend);
        Assert.Equal("CPU",new StoriesExecutionPlan("key",cpu,[10,10,10,10,10]).SelectedBackend);
    }

    [Fact]
    public void InvalidIdentityMissingSamplesAndNonfiniteTimingsCannotBeReused()
    {
        var good=new StoriesExecutionPlan("key",[1,1,1,1,1],[2,2,2,2,2]);
        Assert.True(good.IsValidFor("key"));Assert.False(good.IsValidFor("changed-driver-or-kernel"));
        foreach(double bad in new[] {0,-1,double.NaN,double.PositiveInfinity})
            Assert.False((good with {CpuSamplesMs=[1,1,bad,1,1]}).IsValidFor("key"));
        Assert.False((good with {VulkanSamplesMs=[]}).IsValidFor("key"));
        Assert.False(JsonSerializer.Deserialize<StoriesExecutionPlan>("{\"Identity\":\"key\"}")!.IsValidFor("key"));
        Assert.Equal(good.SelectedBackend,JsonSerializer.Deserialize<StoriesExecutionPlan>(JsonSerializer.Serialize(good))!.SelectedBackend);
    }
}

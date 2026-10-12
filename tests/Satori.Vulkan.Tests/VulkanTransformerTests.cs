using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Core;
using Satori.Validation;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanTransformerTests
{
    private static VulkanContext Create()
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        Assert.True(VulkanContext.TryCreate(new(Validation:true),out var context,out var availability),availability.ToString());
        Assert.False(context!.Info.Software);return context;
    }
    private static JsonDocument Fixture() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,"../../../../../fixtures/operators/transformer-v1.json")));
    private static float[] Input(JsonDocument fixture,string key) => fixture.RootElement.GetProperty(key).EnumerateArray().Select(v=>v.GetSingle()).ToArray();
    private static void Compare(ReadOnlySpan<float> actual,ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length,actual.Length);
        for(int i=0;i<actual.Length;i++)Assert.True(float.IsFinite(actual[i])&&Math.Abs(actual[i]-expected[i])<=2e-6+2e-6*Math.Abs(expected[i]),$"Element {i}: {actual[i]} vs {expected[i]}");
    }
    [Fact]
    public void RmsNormMatchesIndependentFixtureZeroAndProfileRange()
    {
        using var context=Create();using var fixture=Fixture();
        float[] x=Input(fixture,"x"),output=new float[x.Length];
        using(var ops=new VulkanTransformerOperators(context,Input(fixture,"weights")))
        { ops.RmsNorm(x,output);Compare(output,Input(fixture,"rms"));ops.RmsNorm(new float[x.Length],output);Assert.All(output,v=>Assert.Equal(0,v)); }
        float[] weights=Enumerable.Range(0,288).Select(i=>.5f+i/288f).ToArray();
        using(var ops=new VulkanTransformerOperators(context,weights))
        foreach(float magnitude in new[] {1e-8f,1f,100f})
        {
            x=Enumerable.Range(0,288).Select(i=>(i%17-8)*magnitude).ToArray();output=new float[288];float[] expected=new float[288];
            TransformerOps.RmsNorm(x,weights,1e-5f,expected);ops.RmsNorm(x,output);Compare(output,expected);
            var sample=Q8Validation.CriticalCases().First(c=>c.Shape.Columns==288);
            using var matrix=context.CreateQ8Matrix(sample.Payload,sample.Shape,new(Kernel:VulkanQ8Kernel.Cooperative64));
            float[] projected=new float[sample.Shape.Rows];matrix.Execute(output,projected);
            Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,output,projected).Passed);
        }
        context.Dispose();Assert.Equal(0,context.ValidationErrorCount);
    }
}

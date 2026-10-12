using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Core;
using Satori.Validation;
using Satori.Formats.Gguf;

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
    [Fact]
    public void RopeMatchesIndependentFixtureAndContextBoundaryPositions()
    {
        using var context=Create();using var fixture=Fixture();
        using var ops=new VulkanTransformerOperators(context,Enumerable.Repeat(1f,288).ToArray());
        float[] x=Input(fixture,"x");ops.Rope(x,37);Compare(x,Input(fixture,"rope"));
        foreach(int position in new[] {0,1,31,63,127})
        {
            x=Enumerable.Range(0,288).Select(i=>(i%31-15)/8f).ToArray();var expected=x.ToArray();
            TransformerOps.Rope(expected,48,position);ops.Rope(x,position);Compare(x,expected);
        }
        context.Dispose();Assert.Equal(0,context.ValidationErrorCount);
    }
    [Fact]
    public void SwiGluMatchesFixtureExtremesAndRealFeedForwardBlock()
    {
        using var context=Create();using var fixture=Fixture();
        using var file=new GgufFile(Path.Combine(AppContext.BaseDirectory,"../../../../../.temp/gguf-corpus/stories15M-q8_0.gguf"));
        var weights=new GgufFloatVector(file,"blk.0.ffn_norm.weight");
        using var ops=new VulkanTransformerOperators(context,weights.Values);
        float[] gate=Input(fixture,"gate"),up=Input(fixture,"up"),actual=new float[gate.Length];
        ops.SwiGlu(gate,up,actual);Compare(actual,Input(fixture,"swiglu"));
        gate=[-1000,-100,-10,-1,0,1,10,1000];up=Enumerable.Repeat(1f,8).ToArray();
        float[] expected=new float[8];TransformerOps.SwiGlu(gate,up,expected);ops.SwiGlu(gate,up,gate);Compare(gate,expected);
        var g=new GgufQuantizedMatrix(file,"blk.0.ffn_gate.weight");var u=new GgufQuantizedMatrix(file,"blk.0.ffn_up.weight");var d=new GgufQuantizedMatrix(file,"blk.0.ffn_down.weight");
        using var gg=g.CreateExecution(context);using var ug=u.CreateExecution(context);using var dg=d.CreateExecution(context);
        float[] x=Enumerable.Range(0,288).Select(i=>(i%17-8)/16f).ToArray(),norm=new float[288],cpuNorm=new float[288];
        ops.RmsNorm(x,norm);TransformerOps.RmsNorm(x,weights.Values,1e-5f,cpuNorm);
        gate=new float[768];up=new float[768];float[] cg=new float[768],cu=new float[768];
        gg.Execute(norm,gate);ug.Execute(norm,up);g.Execute(cpuNorm,cg);u.Execute(cpuNorm,cu);
        ops.SwiGlu(gate,up,gate);TransformerOps.SwiGlu(cg,cu,cg);
        actual=new float[288];expected=new float[288];dg.Execute(gate,actual);d.Execute(cg,expected);Compare(actual,expected);
        context.Dispose();Assert.Equal(0,context.ValidationErrorCount);
    }
}

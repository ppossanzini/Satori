using Satori.Core;

namespace Satori.Numerics.Tests;

public sealed class TokenSamplerTests
{
    [Fact]
    public void SeededUniformSequenceMatchesIndependentSplitMix64Vectors()
    {
        // Seed zero: reference high bits of the first five SplitMix64 outputs.
        var sampler = new TokenSampler(4,new(1));
        Assert.Equal(new[] {3,1,0,3,0},Enumerable.Range(0,5).Select(_=>sampler.Select([0,0,0,0])).ToArray());
    }

    [Fact]
    public void DistributionMatchesAnalyticalSoftmaxAfterTemperatureAndFilters()
    {
        // exp(log(4)/2):exp(log(9)/2):exp(0) = 2:3:1.
        var sampler = new TokenSampler(3,new(2,Seed:42));
        int[] counts = new int[3];
        float[] logits = [(float)Math.Log(4),(float)Math.Log(9),0];
        for (int i = 0; i < 60000; i++) counts[sampler.Select(logits)]++;
        Assert.InRange(counts[0],19300,20700);
        Assert.InRange(counts[1],29300,30700);
        Assert.InRange(counts[2],9300,10700);
        var filtered = new TokenSampler(3,new(2,2,0.59,42));
        for (int i = 0; i < 1000; i++) Assert.Equal(1,filtered.Select(logits));
        var nucleus = new TokenSampler(3,new(2,0,0.8,42));
        Array.Clear(counts);
        for (int i = 0; i < 50000; i++) counts[nucleus.Select(logits)]++;
        Assert.Equal(0,counts[2]);
        Assert.InRange(counts[0],19300,20700);
        Assert.InRange(counts[1],29300,30700);
        Assert.Equal(new[] {(float)Math.Log(4),(float)Math.Log(9),0},logits);
    }

    [Fact]
    public void LimitsTiesAndExtremeTemperaturesStayFiniteAndDeterministic()
    {
        Assert.Equal(0,new TokenSampler(3,new(1,1)).Select([2,2,1]));
        Assert.Equal(0,new TokenSampler(3,new(1,0,0.5)).Select([0,0,-float.MaxValue]));
        Assert.Equal(1,new TokenSampler(3,new(double.Epsilon)).Select([-float.MaxValue,float.MaxValue,0]));
        Assert.Equal(1,new TokenSampler(3).Select([0,1,1]));
        var a = new TokenSampler(3,new(0.7,99,0.9,ulong.MaxValue));
        var b = new TokenSampler(3,new(0.7,99,0.9,ulong.MaxValue));
        for (int i = 0; i < 100; i++) Assert.Equal(a.Select([1,2,3]),b.Select([1,2,3]));
        Assert.InRange(new TokenSampler(3,new(double.MaxValue)).Select([float.MaxValue,0,-float.MaxValue]),0,2);
    }

    [Fact]
    public void InvalidConfigurationAndLogitsAreRejectedEvenInGreedyMode()
    {
        foreach (var options in new[] {new SamplingOptions(-1),new(double.NaN),new(double.PositiveInfinity),new(TopK:-1),new(TopP:0),new(TopP:1.1),new(TopP:double.NaN)})
            Assert.Throws<ArgumentOutOfRangeException>(()=>new TokenSampler(3,options));
        var sampler = new TokenSampler(3);
        Assert.Throws<ArgumentException>(()=>sampler.Select([1]));
        Assert.Throws<InvalidDataException>(()=>sampler.Select([1,float.NaN,0]));
    }
}

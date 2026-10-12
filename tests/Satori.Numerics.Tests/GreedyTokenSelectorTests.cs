using Satori.Core;

namespace Satori.Numerics.Tests;

public sealed class GreedyTokenSelectorTests
{
    [Fact]
    public void ArgmaxUsesSmallestIdAcrossTiesAndFiniteExtremes()
    {
        Assert.Equal(0,GreedyTokenSelector.Select([-0f,0f,-float.Epsilon]));
        Assert.Equal(1,GreedyTokenSelector.Select([-float.MaxValue,float.MaxValue,float.MaxValue]));
        Assert.Equal(2,GreedyTokenSelector.Select([-2,-1,0]));
        Assert.Equal(0,GreedyTokenSelector.Select([-3,-3,-4]));
    }

    [Fact]
    public void NonFiniteValuesAndAbsentLogitsAreRejected()
    {
        Assert.Throws<ArgumentException>(()=>GreedyTokenSelector.Select([]));
        foreach (float invalid in new[] {float.NaN,float.PositiveInfinity,float.NegativeInfinity})
            Assert.Throws<InvalidDataException>(()=>GreedyTokenSelector.Select([1,invalid]));
    }
}

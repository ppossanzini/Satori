using Satori.Validation;

namespace Satori.Numerics.Tests;

public class NumericalCorpusTests
{
    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void KernelStaysWithinRoundingBoundAcrossCriticalCorpus(int blockSize)
    {
        foreach (var sample in NumericalCorpus.CriticalCases(blockSize))
        {
            var result = NumericalValidation.Evaluate(sample, blockSize);
            Assert.True(result.Passed, $"{sample.Name}: error/bound={result.MaxErrorToBoundRatio}");
            if (sample.Name == "random")
                Assert.True(result.OutputQuantizationRms > result.KernelRms * 100);
        }
    }
}

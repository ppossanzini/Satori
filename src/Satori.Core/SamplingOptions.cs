namespace Satori.Core;

public sealed record SamplingOptions(double Temperature = 0, int TopK = 0, double TopP = 1, ulong Seed = 0)
{
    public void Validate()
    {
        if (!double.IsFinite(Temperature) || Temperature < 0) throw new ArgumentOutOfRangeException(nameof(Temperature));
        if (TopK < 0) throw new ArgumentOutOfRangeException(nameof(TopK));
        if (!double.IsFinite(TopP) || TopP <= 0 || TopP > 1) throw new ArgumentOutOfRangeException(nameof(TopP));
    }
}

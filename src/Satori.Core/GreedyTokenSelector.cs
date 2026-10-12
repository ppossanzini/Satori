namespace Satori.Core;

/// <summary>Selects the smallest ID having the maximum finite logit.</summary>
public static class GreedyTokenSelector
{
    public static int Select(ReadOnlySpan<float> logits)
    {
        if (logits.IsEmpty) throw new ArgumentException("Logits are empty.",nameof(logits));
        int best = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (!float.IsFinite(logits[i])) throw new InvalidDataException("Non-finite logit.");
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }
}

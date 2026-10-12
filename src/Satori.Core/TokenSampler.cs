namespace Satori.Core;

/// <summary>Stateful seeded sampling. Each instance belongs to one generation; not thread safe.</summary>
public sealed class TokenSampler
{
    private readonly SamplingOptions _options;
    private readonly int[] _ids;
    private readonly double[] _weights;
    private readonly IComparer<int> _order;
    private ulong _state;

    public TokenSampler(int vocabularySize, SamplingOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabularySize);
        _options = options ?? new();
        _options.Validate();
        _state = _options.Seed;
        _ids = new int[vocabularySize];
        _weights = new double[vocabularySize];
        _order = Comparer<int>.Create((a,b) =>
        {
            int comparison = _weights[b].CompareTo(_weights[a]);
            return comparison != 0 ? comparison : a.CompareTo(b);
        });
    }

    public int Select(ReadOnlySpan<float> logits)
    {
        if (logits.Length != _ids.Length) throw new ArgumentException("Logits must match the vocabulary size.",nameof(logits));
        int maximum = GreedyTokenSelector.Select(logits);
        if (_options.Temperature == 0) return maximum;
        for (int i = 0; i < logits.Length; i++)
        {
            _ids[i] = i;
            _weights[i] = logits[i];
        }
        Array.Sort(_ids,_order);
        for (int i = 0; i < logits.Length; i++)
            _weights[i] = Math.Exp(((double)logits[i]-logits[maximum])/_options.Temperature);
        int count = _options.TopK == 0 ? logits.Length : Math.Min(_options.TopK,logits.Length);
        double total = 0;
        for (int i = 0; i < count; i++) total += _weights[_ids[i]];
        double retained = 0;
        int kept = 0;
        do { retained += _weights[_ids[kept++]]; }
        while (kept < count && retained < _options.TopP*total);
        double target = NextUnit()*retained;
        double cumulative = 0;
        for (int i = 0; i < kept; i++)
        {
            cumulative += _weights[_ids[i]];
            if (target < cumulative) return _ids[i];
        }
        return _ids[kept-1];
    }

    // SplitMix64, with an explicit 53-bit conversion to [0,1).
    private double NextUnit()
    {
        unchecked
        {
            ulong z = (_state += 0x9E3779B97F4A7C15UL);
            z = (z^(z>>30))*0xBF58476D1CE4E5B9UL;
            z = (z^(z>>27))*0x94D049BB133111EBUL;
            z ^= z>>31;
            return (z>>11)*(1.0/9007199254740992.0);
        }
    }
}

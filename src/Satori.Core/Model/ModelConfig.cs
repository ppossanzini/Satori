using System.Globalization;

namespace Satori.Core.Model;

/// <summary>
/// Typed read-only view over the scalar metadata of a model. Keys are addressed by suffix and resolved
/// against the architecture prefix (for example <c>block_count</c> becomes <c>qwen3.block_count</c>).
/// A missing required key is reported by the caller as a capability gap, never defaulted.
/// </summary>
public sealed class ModelConfig
{
    private readonly IReadOnlyDictionary<string, object?> _values;

    public ModelConfig(string architecture, IReadOnlyDictionary<string, object?> values)
    {
        Architecture = architecture;
        _values = values;
    }

    public string Architecture { get; }

    public bool TryInt(string suffix, out long value)
    {
        value = 0;
        if (!_values.TryGetValue(Full(suffix), out var raw) || raw is null) return false;
        switch (raw)
        {
            case byte v: value = v; return true;
            case sbyte v: value = v; return true;
            case ushort v: value = v; return true;
            case short v: value = v; return true;
            case uint v: value = v; return true;
            case int v: value = v; return true;
            case ulong v when v <= long.MaxValue: value = (long)v; return true;
            case long v: value = v; return true;
            case float v when v == MathF.Floor(v) && v >= long.MinValue && v <= long.MaxValue:
                value = (long)v; return true;
            case double v when v == Math.Floor(v) && v >= long.MinValue && v <= long.MaxValue:
                value = (long)v; return true;
            case string text when long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long parsed):
                value = parsed; return true;
            default: return false;
        }
    }

    public bool TryFloat(string suffix, out double value)
    {
        value = 0;
        if (!_values.TryGetValue(Full(suffix), out var raw) || raw is null) return false;
        switch (raw)
        {
            case float v: value = v; return true;
            case double v: value = v; return true;
            case byte v: value = v; return true;
            case sbyte v: value = v; return true;
            case ushort v: value = v; return true;
            case short v: value = v; return true;
            case uint v: value = v; return true;
            case int v: value = v; return true;
            case ulong v: value = v; return true;
            case long v: value = v; return true;
            case string text when double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed):
                value = parsed; return true;
            default: return false;
        }
    }

    public bool TryString(string suffix, out string value)
    {
        if (_values.TryGetValue(Full(suffix), out var raw) && raw is string text)
        {
            value = text;
            return true;
        }
        value = string.Empty;
        return false;
    }

    public bool TryBool(string suffix, out bool value)
    {
        if (_values.TryGetValue(Full(suffix), out var raw) && raw is bool flag)
        {
            value = flag;
            return true;
        }
        value = false;
        return false;
    }

    private string Full(string suffix) => $"{Architecture}.{suffix}";
}

namespace Satori.Core;

internal readonly ref struct Ternary2Execution(
    Ternary2Shape shape, ReadOnlySpan<float> input, Span<float> output, bool useAvx2)
{
    internal Ternary2Shape Shape { get; } = shape;
    internal ReadOnlySpan<float> Input { get; } = input;
    internal Span<float> Output { get; } = output;
    internal bool UseAvx2 { get; } = useAvx2;
}

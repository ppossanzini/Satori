namespace Satori.Core;

public sealed class QuantizedMatrix
{
    private readonly byte[] _data;
    public QuantizedMatrixShape Shape { get; }

    public QuantizedMatrix(ReadOnlySpan<byte> payload, QuantizedMatrixShape shape)
    {
        Shape = shape;
        _data = payload.ToArray();
        QuantizedMatVec.ValidatePayload(_data, shape);
    }

    public void Execute(ReadOnlySpan<float> input, Span<float> output) =>
        QuantizedMatVec.Execute(_data, Shape, input, output);

    public void ExecuteScalar(ReadOnlySpan<float> input, Span<float> output) =>
        QuantizedMatVec.ExecuteScalar(_data, Shape, input, output);
}

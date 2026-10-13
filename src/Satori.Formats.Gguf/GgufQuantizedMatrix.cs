using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>A validated numerical tensor borrowing its file's lifetime.</summary>
public sealed class GgufQuantizedMatrix
{
    private readonly GgufFile _file;
    private readonly string _name;
    public string Name => _name;
    public QuantizedMatrixShape Shape { get; }

    public GgufQuantizedMatrix(GgufFile file, string name)
    {
        var tensor = file.Tensors[name];
        if (tensor.Type is not (8 or 2 or 12 or 14) || tensor.Dimensions.Count != 2 || tensor.ByteLength > int.MaxValue ||
            tensor.Dimensions.Any(d => d > int.MaxValue))
            throw new NotSupportedException("MatVec requires a rank-2 Q8_0/Q4_0/Q4_K/Q6_K tensor within Int32 limits.");
        Shape = new((int)tensor.Dimensions[1], (int)tensor.Dimensions[0], tensor.Type switch
        {
            8 => BlockQuantization.Q8_0,
            2 => BlockQuantization.Q4_0,
            12 => BlockQuantization.Q4_K,
            _ => BlockQuantization.Q6_K,
        });
        _file = file;
        _name = name;
        file.ReadTensor(name, 0, Shape.ByteLength, static (bytes, shape) => QuantizedMatVec.ValidatePayload(bytes, shape), Shape);
    }

    public IQuantizedMatrix CreateExecution(IQuantizedMatrixFactory factory)
    {
        IQuantizedMatrix? execution = null;
        _file.ReadTensor(_name,0,Shape.ByteLength,(bytes,owner)=>execution=owner.Create(bytes,Shape),factory);
        return execution!;
    }

    public void Execute(ReadOnlySpan<float> input, Span<float> output) =>
        _file.ReadTensor(_name, 0, Shape.ByteLength, static (bytes, state) =>
            QuantizedMatVec.Execute(bytes, state.Shape, state.Input, state.Output), new Execution(Shape, input, output));

    public void ExecuteScalar(ReadOnlySpan<float> input, Span<float> output) =>
        _file.ReadTensor(_name, 0, Shape.ByteLength, static (bytes, state) =>
            QuantizedMatVec.ExecuteScalar(bytes, state.Shape, state.Input, state.Output), new Execution(Shape, input, output));

    public void ReadRow(int row, Span<float> output)
    {
        if (Shape.Format != BlockQuantization.Q8_0) throw new NotSupportedException("Embedding requires Q8_0.");
        if ((uint)row >= (uint)Shape.Rows || output.Length != Shape.Columns) throw new ArgumentOutOfRangeException(nameof(row));
        int rowBytes = Shape.ByteLength / Shape.Rows;
        _file.ReadTensor(_name, checked((long)row * rowBytes), rowBytes,
            static (bytes, destination) => QuantizedRow.DecodeQ8(bytes, destination), output);
    }

    private readonly ref struct Execution(QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        public QuantizedMatrixShape Shape { get; } = shape;
        public ReadOnlySpan<float> Input { get; } = input;
        public Span<float> Output { get; } = output;
    }
}

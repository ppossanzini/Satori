namespace Satori.Core;

/// <summary>Validated Ternary2 file executed directly from read-only mapped storage.</summary>
public sealed class MappedTernary2Matrix : IDisposable
{
    private readonly MappedWeights _weights;
    private readonly Ternary2Shape _shape;

    public MappedTernary2Matrix(string path)
    {
        _weights = new MappedWeights(path);
        try
        {
            if (_weights.Length < Ternary2FileFormat.HeaderSize)
                throw new InvalidDataException("Truncated Ternary2 header.");
            Span<Ternary2Shape> header = stackalloc Ternary2Shape[1];
            _weights.Read(0, Ternary2FileFormat.HeaderSize,
                static (bytes, state) => state[0] = Ternary2FileFormat.ReadHeader(bytes), header);
            _shape = header[0];
            if (_weights.Length != (long)Ternary2FileFormat.HeaderSize + _shape.ByteLength)
                throw new InvalidDataException("Truncated payload or trailing file data.");
            _weights.Read(Ternary2FileFormat.HeaderSize, _shape.ByteLength,
                static (bytes, shape) => Ternary2Matrix.ValidatePayload(shape, bytes), _shape);
        }
        catch (Exception exception) when (exception is ArgumentException or OverflowException)
        {
            _weights.Dispose();
            throw new InvalidDataException("Invalid Ternary2 shape or payload.", exception);
        }
        catch
        {
            _weights.Dispose();
            throw;
        }
    }

    public int Rows => _shape.Rows;
    public int Columns => _shape.Columns;
    public int BlockSize => _shape.BlockSize;
    public int ByteLength => _shape.ByteLength;

    /// <summary>Executes AVX2 when available, otherwise scalar, without copying mapped weights.</summary>
    public void Execute(ReadOnlySpan<float> input, Span<float> output) => ExecuteCore(input, output, useAvx2: true);

    public void ExecuteScalar(ReadOnlySpan<float> input, Span<float> output) => ExecuteCore(input, output, useAvx2: false);

    private void ExecuteCore(ReadOnlySpan<float> input, Span<float> output, bool useAvx2)
    {
        Ternary2MatVec.ValidateVectors(_shape, input, output);
        _weights.Read(Ternary2FileFormat.HeaderSize, _shape.ByteLength,
            static (bytes, state) => Ternary2MatVec.ExecuteData(bytes, state),
            new Ternary2Execution(_shape, input, output, useAvx2));
    }

    public void Dispose() => _weights.Dispose();
}

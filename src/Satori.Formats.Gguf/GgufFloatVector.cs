using System.Buffers.Binary;

namespace Satori.Formats.Gguf;

/// <summary>Small owned F32 vector, decoded and validated once from little-endian GGUF.</summary>
public sealed class GgufFloatVector
{
    private readonly float[] _values;
    public ReadOnlySpan<float> Values => _values;

    public GgufFloatVector(GgufFile file, string name)
    {
        var tensor = file.Tensors[name];
        if (tensor.Type != 0 || tensor.Dimensions.Count != 1 || tensor.ByteLength > int.MaxValue)
            throw new NotSupportedException("Requires a rank-1 F32 tensor within Int32 limits.");
        _values = new float[checked((int)tensor.Dimensions[0])];
        file.ReadTensor(name, 0, checked(_values.Length * 4), static (bytes, values) =>
        {
            for (int i = 0; i < values.Length; i++)
            {
                float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice(i * 4, 4)));
                if (!float.IsFinite(value)) throw new InvalidDataException("Non-finite F32 vector.");
                values[i] = value;
            }
        }, _values);
    }
}

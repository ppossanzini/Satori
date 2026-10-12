using System.Buffers;
using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.Text;
using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>Validated GGUF catalogue with callback-scoped access to immutable mapped weights.</summary>
public sealed class GgufFile : IDisposable
{
    private readonly MappedWeights _mapping;
    private readonly Dictionary<string, GgufTensor> _tensors = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgufMetadata> _metadata = new(StringComparer.Ordinal);
    private static readonly UTF8Encoding Utf8 = new(false, true);

    public GgufFile(string path)
    {
        long fileLength = new FileInfo(path).Length;
        if (fileLength == 0) throw new InvalidDataException("GGUF at byte 0: empty file.");
        if (fileLength > 1L << 40) throw new NotSupportedException("File limit: 1 TiB.");
        _mapping = new MappedWeights(path);
        try
        {
            if (_mapping.Length > 1L << 40) throw new NotSupportedException("File limit: 1 TiB.");
            Parse();
            Tensors = new ReadOnlyDictionary<string, GgufTensor>(_tensors);
            Metadata = new ReadOnlyDictionary<string, GgufMetadata>(_metadata);
        }
        catch (OverflowException exception)
        {
            _mapping.Dispose();
            throw new InvalidDataException("GGUF: offset or dimension overflow.", exception);
        }
        catch { _mapping.Dispose(); throw; }
    }

    public IReadOnlyDictionary<string, GgufTensor> Tensors { get; }
    public IReadOnlyDictionary<string, GgufMetadata> Metadata { get; }
    public long DataOffset { get; private set; }

    public void ReadTensor<TState>(string name, long offset, int length, ReadOnlySpanAction<byte, TState> action, TState state)
        where TState : allows ref struct
    {
        var tensor = _tensors[name];
        if (offset < 0 || length < 0 || offset > tensor.ByteLength || length > tensor.ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset), "Window exceeds tensor payload.");
        _mapping.Read(checked(tensor.Offset + offset), length, action, state);
    }

    public void ReadMetadata<TState>(string key, ReadOnlySpanAction<byte, TState> action, TState state)
        where TState : allows ref struct
    {
        var entry = _metadata[key];
        if (entry.ByteLength > int.MaxValue) throw new NotSupportedException("Metadata window exceeds Int32.");
        _mapping.Read(entry.Offset, (int)entry.ByteLength, action, state);
    }

    public object ReadScalar(string key)
    {
        var entry = _metadata[key];
        if (entry.Type == 9) throw new InvalidOperationException("Use scoped access for arrays.");
        object?[] result = new object?[1];
        ReadMetadata(key, (bytes, state) => state[0] = entry.Type switch
        {
            0 => bytes[0], 1 => (sbyte)bytes[0],
            2 => BinaryPrimitives.ReadUInt16LittleEndian(bytes), 3 => BinaryPrimitives.ReadInt16LittleEndian(bytes),
            4 => BinaryPrimitives.ReadUInt32LittleEndian(bytes), 5 => BinaryPrimitives.ReadInt32LittleEndian(bytes),
            6 => BinaryPrimitives.ReadSingleLittleEndian(bytes), 7 => bytes[0] != 0,
            8 => Utf8.GetString(bytes[8..]),
            10 => BinaryPrimitives.ReadUInt64LittleEndian(bytes), 11 => BinaryPrimitives.ReadInt64LittleEndian(bytes),
            12 => BinaryPrimitives.ReadDoubleLittleEndian(bytes),
            _ => throw new InvalidOperationException()
        }, result);
        return result[0]!;
    }

    private void Parse()
    {
        var cursor = new GgufCursor(_mapping);
        if (cursor.Unsigned(4) != 0x46554747) throw cursor.Error("Invalid magic");
        if (cursor.Unsigned(4) != 3) throw new NotSupportedException("Only little-endian GGUF v3 is supported.");
        long tensorCount = cursor.Count(cursor.Unsigned(8));
        long metadataCount = cursor.Count(cursor.Unsigned(8));
        if (tensorCount == 0) throw new NotSupportedException("Tensor count zero is outside the profile.");
        if (tensorCount > 100000 || metadataCount > 100000) throw new NotSupportedException("Catalogue count limit: 100000.");
        for (long i = 0; i < metadataCount; i++)
        {
            string key = cursor.String(65535);
            if (!ValidKey(key)) throw cursor.Error("Invalid metadata key");
            uint type = (uint)cursor.Unsigned(4);
            long start = cursor.Position;
            uint? elementType = null;
            long? count = null;
            if (type == 9)
            {
                elementType = (uint)cursor.Unsigned(4);
                count = cursor.Count(cursor.Unsigned(8));
                if (count > 1048576) throw new NotSupportedException("Array count limit: 1048576.");
                CheckType(elementType.Value);
                for (long j = 0; j < count; j++) SkipScalar(cursor, elementType.Value);
            }
            else SkipScalar(cursor, type);
            if (!_metadata.TryAdd(key, new(key, type, start, cursor.Position - start, elementType, count)))
                throw cursor.Error("Duplicate metadata key");
        }
        uint alignment = Get<uint>("general.alignment", 4, 32);
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) throw cursor.Error("Invalid alignment");
        if (alignment < 8 || alignment > 4096) throw new NotSupportedException("Alignment range: 8 to 4096.");
        string architecture = Get<string>("general.architecture", 8, "");
        if (architecture.Length == 0 || !architecture.All(c => c is >= 'a' and <= 'z' or >= '0' and <= '9'))
            throw cursor.Error("Missing or invalid general.architecture");
        if (Get<string>("general.type", 8, "model") != "model") throw new NotSupportedException("Only model files are supported.");
        _ = Get<uint>("general.file_type", 4, 0);
        if (Get<ushort>("split.count", 2, 1) != 1 || Get<ushort>("split.no", 2, 0) != 0)
            throw new NotSupportedException("Sharded GGUF is unsupported.");
        int splitCount = Get<int>("split.tensors.count", 5, (int)tensorCount);
        if (splitCount != tensorCount) throw cursor.Error("Invalid split.tensors.count");
        bool quantized = false;
        long expectedOffset = 0;
        var relative = new List<GgufTensor>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        for (long i = 0; i < tensorCount; i++)
        {
            string name = cursor.String(63);
            if (name.Length == 0 || name.Contains('\0')) throw cursor.Error("Invalid tensor name");
            uint rank = (uint)cursor.Unsigned(4);
            if (rank == 0) throw cursor.Error("Zero rank");
            if (rank > 4) throw new NotSupportedException("Tensor rank limit: 4.");
            var dimensions = new long[rank];
            long elements = 1;
            for (int d = 0; d < rank; d++)
            {
                dimensions[d] = cursor.Count(cursor.Unsigned(8));
                if (dimensions[d] == 0) throw cursor.Error("Zero dimension");
                elements = checked(elements * dimensions[d]);
            }
            uint type = (uint)cursor.Unsigned(4);
            (int block, int bytes) = type switch
            {
                0 => (1, 4), 1 => (1, 2), 2 => (32, 18), 8 => (32, 34),
                _ => throw new NotSupportedException($"Tensor type {type} is unsupported.")
            };
            quantized |= block == 32;
            if (dimensions[0] % block != 0) throw cursor.Error("Quantized row is not a multiple of 32");
            long length = checked(elements / block * bytes);
            long offset = cursor.Count(cursor.Unsigned(8));
            if (offset % alignment != 0) throw cursor.Error("Unaligned tensor offset");
            _ = checked(offset + Align(length, alignment));
            if (!names.Add(name)) throw cursor.Error("Duplicate tensor name");
            relative.Add(new(name, dimensions, type, offset, length));
        }
        if (quantized && !_metadata.ContainsKey("general.quantization_version")) throw cursor.Error("Missing quantization version");
        if (Get<uint>("general.quantization_version", 4, 2) != 2) throw new NotSupportedException("Quantization version must be 2.");
        DataOffset = Align(cursor.Position, alignment);
        cursor.Skip(DataOffset - cursor.Position);
        long previousEnd = 0;
        foreach (var tensor in relative.OrderBy(t => t.Offset))
        {
            long end = checked(tensor.Offset + Align(tensor.ByteLength, alignment));
            if (end > _mapping.Length - DataOffset) throw cursor.Error("Tensor data or final padding exceeds file");
            if (tensor.Offset < previousEnd) throw cursor.Error("Overlapping tensor ranges");
            previousEnd = end;
        }
        foreach (var tensor in relative)
        {
            if (tensor.Offset != expectedOffset) throw new NotSupportedException("Noncanonical tensor gap/order.");
            expectedOffset = checked(tensor.Offset + Align(tensor.ByteLength, alignment));
        }
        foreach (var tensor in relative)
            _tensors.Add(tensor.Name, new(tensor.Name, tensor.Dimensions.ToArray(), tensor.Type,
                checked(DataOffset + tensor.Offset), tensor.ByteLength));
    }

    private T Get<T>(string key, uint type, T fallback)
    {
        if (!_metadata.TryGetValue(key, out var entry)) return fallback;
        if (entry.Type != type) throw new InvalidDataException($"GGUF at byte {entry.Offset}: wrong type for {key}.");
        return (T)ReadScalar(key);
    }

    private static bool ValidKey(string key) => key.Length > 0 && key.Split('.').All(part =>
        part.Length > 0 && part.All(c => c is >= 'a' and <= 'z' or >= '0' and <= '9' or '_'));

    private static void CheckType(uint type)
    {
        if (type > 12 || type == 9) throw new NotSupportedException($"Metadata scalar type {type} is unsupported.");
    }

    private static void SkipScalar(GgufCursor cursor, uint type)
    {
        CheckType(type);
        if (type == 8) { _ = cursor.String(16 * 1024 * 1024); return; }
        if (type == 7)
        {
            if (cursor.Unsigned(1) > 1) throw cursor.Error("Invalid Boolean");
            return;
        }
        cursor.Skip(type switch { 0 or 1 => 1, 2 or 3 => 2, 4 or 5 or 6 => 4, _ => 8 });
    }

    private static long Align(long value, uint alignment) => checked((value + alignment - 1) & ~((long)alignment - 1));
    public void Dispose() => _mapping.Dispose();
}

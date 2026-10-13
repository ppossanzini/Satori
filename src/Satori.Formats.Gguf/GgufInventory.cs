using System.Buffers;
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>Directory entry of a tensor as declared by the container, with computed byte length when the type is catalogued.</summary>
public sealed record GgufTensorDescriptor(
    string Name, uint TypeId, string TypeName, bool Catalogued, bool Supported,
    IReadOnlyList<long> Dimensions, long Offset, long ByteLength, bool LengthKnown);

/// <summary>Metadata entry with its declared type and decoded scalar value (arrays are described, not materialised).</summary>
public sealed record GgufMetadataDescriptor(
    string Key, uint Type, string TypeName, uint? ElementType, string? ElementTypeName, long? ElementCount, object? Value);

/// <summary>
/// Read-only descriptor inventory of a GGUF container: header, metadata and tensor directory, without
/// decoding weights or executing anything. Serves the capability matrix (<c>DS02</c>) and the reader
/// the inventory command (<c>DS01</c>).
/// </summary>
/// <remarks>
/// Unlike <see cref="GgufFile"/>, which rejects a file outside the execution profile, this scan is
/// tolerant: it reports unsupported tensor types and never silently drops a declared entry. Bytes are
/// mapped through <see cref="MappedWeights"/>; only header, metadata and directory regions are touched.
/// </remarks>
public sealed record GgufInventory(
    string Path, long FileLength, uint Version, long TensorCount, long MetadataCount,
    uint Alignment, long DataOffset, long TotalTensorBytes, bool TotalTensorBytesExact,
    string Architecture, IReadOnlyList<string> UnsupportedTypes,
    IReadOnlyList<GgufTensorDescriptor> Tensors, IReadOnlyList<GgufMetadataDescriptor> Metadata)
{
    private static readonly UTF8Encoding Utf8 = new(false, true);

    public bool ProfileCompatible =>
        Version == 3 && UnsupportedTypes.Count == 0 && Architecture.Length > 0;

    public static GgufInventory Scan(string path)
    {
        using var mapping = new MappedWeights(path);
        var cursor = new GgufCursor(mapping);
        if ((uint)cursor.Unsigned(4) != 0x46554747) throw cursor.Error("Invalid magic");
        uint version = (uint)cursor.Unsigned(4);
        long tensorCount = cursor.Count(cursor.Unsigned(8));
        long metadataCount = cursor.Count(cursor.Unsigned(8));
        if (tensorCount > 100000) throw new NotSupportedException("Catalogue count limit: 100000.");
        if (metadataCount > 100000) throw new NotSupportedException("Catalogue count limit: 100000.");

        var metadata = new List<GgufMetadataDescriptor>((int)metadataCount);
        for (long i = 0; i < metadataCount; i++)
        {
            string key = cursor.String(65535);
            uint type = (uint)cursor.Unsigned(4);
            long start = cursor.Position;
            uint? elementType = null;
            long? elementCount = null;
            if (type == 9)
            {
                elementType = (uint)cursor.Unsigned(4);
                elementCount = cursor.Count(cursor.Unsigned(8));
                if (elementCount > 1048576) throw new NotSupportedException("Array count limit: 1048576.");
                if (elementType == 9) throw cursor.Error("Nested arrays are unsupported");
                for (long j = 0; j < elementCount; j++) SkipScalar(cursor, elementType.Value);
            }
            else
            {
                SkipScalar(cursor, type);
            }
            long length = cursor.Position - start;
            object? value = type == 9 ? null : DecodeScalar(mapping, start, length, type);
            metadata.Add(new(key, type, GgufTypes.MetadataTypeName(type), elementType,
                elementType is null ? null : GgufTypes.MetadataTypeName(elementType.Value), elementCount, value));
        }

        uint alignment = ReadAlignment(metadata);
        string architecture = FindValue(metadata, "general.architecture") as string ?? "";

        var tensors = new List<GgufTensorDescriptor>((int)tensorCount);
        var unsupported = new List<string>();
        var unsupportedSeen = new HashSet<uint>();
        long totalBytes = 0;
        bool exact = true;
        for (long i = 0; i < tensorCount; i++)
        {
            string name = cursor.String(63);
            uint rank = (uint)cursor.Unsigned(4);
            if (rank is 0 or > 4) throw cursor.Error("Tensor rank outside 1..4");
            var dimensions = new long[rank];
            long elements = 1;
            for (int d = 0; d < rank; d++)
            {
                dimensions[d] = cursor.Count(cursor.Unsigned(8));
                if (dimensions[d] == 0) throw cursor.Error("Zero dimension");
                elements = checked(elements * dimensions[d]);
            }
            uint type = (uint)cursor.Unsigned(4);
            long offset = cursor.Count(cursor.Unsigned(8));
            var info = GgufTypes.Lookup(type);
            long length = 0;
            bool lengthKnown = false;
            if (info.Catalogued && info.ElementsPerBlock > 0 && dimensions[0] % info.ElementsPerBlock == 0)
            {
                length = checked(elements / info.ElementsPerBlock * info.BytesPerBlock);
                lengthKnown = true;
            }
            if (lengthKnown)
            {
                totalBytes = checked(totalBytes + length);
            }
            else
            {
                exact = false;
            }
            if (!info.Supported && unsupportedSeen.Add(type))
                unsupported.Add($"{info.Name}({type})");
            tensors.Add(new(name, type, info.Name, info.Catalogued, info.Supported, dimensions, offset, length, lengthKnown));
        }

        long dataOffset = Align(cursor.Position, alignment);
        return new GgufInventory(path, mapping.Length, version, tensorCount, metadataCount, alignment,
            dataOffset, totalBytes, exact, architecture, unsupported, tensors, metadata);
    }

    /// <summary>Deterministic text report suitable for a terminal or a reproducible log line.</summary>
    public string ToReport()
    {
        var text = new StringBuilder();
        text.AppendLine($"GGUF inventory: {Path}");
        text.AppendLine($"File bytes: {FileLength}");
        text.AppendLine($"Version: {Version}");
        text.AppendLine($"Architecture: {(Architecture.Length == 0 ? "(missing)" : Architecture)}");
        text.AppendLine($"Tensors: {TensorCount}");
        text.AppendLine($"Metadata: {MetadataCount}");
        text.AppendLine($"Alignment: {Alignment}");
        text.AppendLine($"Data offset: {DataOffset}");
        text.AppendLine($"Tensor data bytes: {TotalTensorBytes}{(TotalTensorBytesExact ? "" : " (partial; unknown-length types present)")}");
        text.AppendLine($"Profile compatible: {(ProfileCompatible ? "yes" : "no")}");
        text.AppendLine($"Unsupported tensor types: {(UnsupportedTypes.Count == 0 ? "none" : string.Join(", ", UnsupportedTypes))}");
        text.AppendLine("Metadata:");
        foreach (var entry in Metadata)
        {
            string detail = entry.Type == 9
                ? $"Array[{entry.ElementTypeName} x {entry.ElementCount}]"
                : Format(entry.Value);
            text.AppendLine($"  {entry.Key} = {detail} ({entry.TypeName})");
        }
        text.AppendLine("Tensors:");
        foreach (var tensor in Tensors)
        {
            string dims = string.Join(" x ", tensor.Dimensions);
            string bytes = tensor.LengthKnown ? tensor.ByteLength.ToString() : "unknown";
            text.AppendLine($"  {tensor.Name} | {tensor.TypeName}({tensor.TypeId}) | [{dims}] | offset={tensor.Offset} | bytes={bytes} | {(tensor.Supported ? "supported" : "unsupported")}");
        }
        return text.ToString();
    }

    /// <summary>Machine-readable report for reproducible inventories and diffs.</summary>
    public string ToJson() => JsonSerializer.Serialize(new
    {
        Path,
        FileLength,
        Version,
        Architecture,
        TensorCount,
        MetadataCount,
        Alignment,
        DataOffset,
        TotalTensorBytes,
        TotalTensorBytesExact,
        ProfileCompatible,
        UnsupportedTypes,
        Metadata = Metadata.Select(m => new { m.Key, m.TypeName, m.ElementTypeName, m.ElementCount, Value = m.Type == 9 ? null : m.Value }),
        Tensors = Tensors.Select(t => new
        {
            t.Name, TypeId = t.TypeId, t.TypeName, t.Catalogued, t.Supported,
            t.Dimensions, t.Offset, t.ByteLength, t.LengthKnown,
        }),
    }, new JsonSerializerOptions { WriteIndented = true });

    private static string Format(object? value) => value switch
    {
        null => "null",
        string s => $"\"{s}\"",
        bool b => b ? "true" : "false",
        _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "null",
    };

    private static uint ReadAlignment(List<GgufMetadataDescriptor> metadata)
    {
        var entry = metadata.FirstOrDefault(m => m.Key == "general.alignment");
        if (entry is { Type: 4, Value: uint value } && value != 0 && (value & (value - 1)) == 0)
            return value;
        return 32;
    }

    private static object? FindValue(List<GgufMetadataDescriptor> metadata, string key) =>
        metadata.FirstOrDefault(m => m.Key == key)?.Value;

    private static void SkipScalar(GgufCursor cursor, uint type)
    {
        if (type > 12 || type == 9) throw new NotSupportedException($"Metadata scalar type {type} is unsupported.");
        if (type == 8) { _ = cursor.String(16 * 1024 * 1024); return; }
        if (type == 7)
        {
            if (cursor.Unsigned(1) > 1) throw cursor.Error("Invalid Boolean");
            return;
        }
        cursor.Skip(type switch { 0 or 1 => 1, 2 or 3 => 2, 4 or 5 or 6 => 4, _ => 8 });
    }

    private static object? DecodeScalar(MappedWeights mapping, long offset, long length, uint type)
    {
        if (length > int.MaxValue) throw new NotSupportedException("Metadata window exceeds Int32.");
        object?[] result = new object?[1];
        ReadOnlySpanAction<byte, object?[]> action = (bytes, state) => state[0] = type switch
        {
            0 => bytes[0], 1 => (sbyte)bytes[0],
            2 => BinaryPrimitives.ReadUInt16LittleEndian(bytes), 3 => BinaryPrimitives.ReadInt16LittleEndian(bytes),
            4 => BinaryPrimitives.ReadUInt32LittleEndian(bytes), 5 => BinaryPrimitives.ReadInt32LittleEndian(bytes),
            6 => BinaryPrimitives.ReadSingleLittleEndian(bytes), 7 => bytes[0] != 0,
            8 => Utf8.GetString(bytes[8..]),
            10 => BinaryPrimitives.ReadUInt64LittleEndian(bytes), 11 => BinaryPrimitives.ReadInt64LittleEndian(bytes),
            12 => BinaryPrimitives.ReadDoubleLittleEndian(bytes),
            _ => throw new NotSupportedException($"Metadata scalar type {type} is unsupported."),
        };
        mapping.Read(offset, (int)length, action, result);
        return result[0];
    }

    private static long Align(long value, uint alignment) => checked((value + alignment - 1) & ~((long)alignment - 1));
}

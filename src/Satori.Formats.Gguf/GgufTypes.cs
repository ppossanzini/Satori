namespace Satori.Formats.Gguf;

/// <summary>Block geometry of a GGML tensor type as declared in the tensor directory.</summary>
/// <param name="Id">The on-disk GGML type identifier.</param>
/// <param name="Name">Canonical GGML type name, or <c>unknown</c> when the identifier is not catalogued.</param>
/// <param name="ElementsPerBlock">Elements covered by one block; <c>0</c> when the block size is unknown.</param>
/// <param name="BytesPerBlock">Bytes occupied by one block; <c>0</c> when the block size is unknown.</param>
/// <param name="Catalogued">Whether Satori knows the block geometry and can compute a byte length.</param>
/// <param name="Supported">Whether the current runtime profile can decode this type.</param>
public readonly record struct GgufTypeInfo(
    uint Id, string Name, int ElementsPerBlock, int BytesPerBlock, bool Catalogued, bool Supported);

/// <summary>GGML tensor type catalogue used for inventory and capability reporting.</summary>
/// <remarks>
/// Geometry follows the pinned GGML revision in <c>docs/gguf-subset-v1.md</c>. Private fork types
/// (for example Prism PTQ1_0/PQ2_0) are deliberately absent: their identifiers and layout must be
/// verified against their own reference before being catalogued.
/// </remarks>
public static class GgufTypes
{
    // Supported by the current execution profile (docs/gguf-subset-v1.md, extended by DW06.1).
    private static readonly HashSet<uint> Supported = [0, 1, 2, 8, 12];

    private static readonly Dictionary<uint, GgufTypeInfo> Catalogue = Build();

    public static GgufTypeInfo Lookup(uint id) =>
        Catalogue.TryGetValue(id, out var info) ? info : new(id, "unknown", 0, 0, false, false);

    /// <summary>Human-readable name for a metadata scalar type identifier.</summary>
    public static string MetadataTypeName(uint type) => type switch
    {
        0 => "UInt8",
        1 => "Int8",
        2 => "UInt16",
        3 => "Int16",
        4 => "UInt32",
        5 => "Int32",
        6 => "Float32",
        7 => "Bool",
        8 => "String",
        9 => "Array",
        10 => "UInt64",
        11 => "Int64",
        12 => "Float64",
        _ => "unknown",
    };

    private static Dictionary<uint, GgufTypeInfo> Build()
    {
        // (id, name, elementsPerBlock, bytesPerBlock)
        (uint, string, int, int)[] rows =
        [
            (0, "F32", 1, 4), (1, "F16", 1, 2), (2, "Q4_0", 32, 18), (3, "Q4_1", 32, 20),
            (6, "Q5_0", 32, 22), (7, "Q5_1", 32, 24), (8, "Q8_0", 32, 34), (9, "Q8_1", 32, 36),
            (10, "Q2_K", 256, 84), (11, "Q3_K", 256, 110), (12, "Q4_K", 256, 144), (13, "Q5_K", 256, 176),
            (14, "Q6_K", 256, 210), (15, "Q8_K", 256, 292),
            (16, "IQ2_XXS", 256, 66), (17, "IQ2_XS", 256, 74), (18, "IQ3_XXS", 256, 98),
            (19, "IQ1_S", 256, 50), (20, "IQ4_NL", 32, 18), (21, "IQ3_S", 256, 110), (22, "IQ2_S", 256, 82),
            (23, "IQ4_XS", 256, 136), (24, "I8", 1, 1), (25, "I16", 1, 2), (26, "I32", 1, 4),
            (27, "I64", 1, 8), (28, "F64", 1, 8), (29, "IQ1_M", 256, 56), (30, "BF16", 1, 2),
            (34, "TQ1_0", 256, 54), (35, "TQ2_0", 256, 66),
        ];
        var map = new Dictionary<uint, GgufTypeInfo>(rows.Length);
        foreach (var (id, name, block, bytes) in rows)
            map[id] = new(id, name, block, bytes, true, Supported.Contains(id));
        return map;
    }
}

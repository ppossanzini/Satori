namespace Satori.Core;

/// <summary>Decodes one validated Q8_0 row without expanding the matrix.</summary>
public static class QuantizedRow
{
    public static void DecodeQ8(ReadOnlySpan<byte> bytes, Span<float> output)
    {
        if (output.Length == 0 || output.Length % 32 != 0 || bytes.Length != checked(output.Length / 32 * 34))
            throw new ArgumentException("Invalid Q8_0 row dimensions.");
        if (bytes.Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Row output overlaps packed input.");
        for (int block = 0; block < output.Length / 32; block++)
        {
            var data = bytes.Slice(block * 34, 34);
            float scale = QuantizedMatVec.Scale(data);
            for (int j = 0; j < 32; j++) output[block * 32 + j] = scale * unchecked((sbyte)data[2 + j]);
        }
    }
}

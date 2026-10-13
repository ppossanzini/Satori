namespace Satori.Core.Tokenization;

/// <summary>GPT-2 byte-level mapping: every byte maps to a printable Unicode character and back.</summary>
internal static class ByteLevel
{
    private static readonly char[] ByteToChar = Build();
    private static readonly Dictionary<char, byte> Map = BuildReverse(ByteToChar);

    public static IReadOnlyDictionary<char, byte> CharToByte => Map;

    /// <summary>Maps the UTF-8 bytes of <paramref name="text"/> to the byte-level alphabet.</summary>
    public static string Encode(string text)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(text);
        var chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) chars[i] = ByteToChar[bytes[i]];
        return new string(chars);
    }

    private static char[] Build()
    {
        var direct = new List<int>();
        for (int b = '!'; b <= '~'; b++) direct.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) direct.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) direct.Add(b);

        var table = new char[256];
        var used = new bool[256];
        foreach (int b in direct)
        {
            table[b] = (char)b;
            used[b] = true;
        }
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!used[b]) table[b] = (char)(256 + extra++);
        }
        return table;
    }

    private static Dictionary<char, byte> BuildReverse(char[] table)
    {
        var map = new Dictionary<char, byte>(256);
        for (int b = 0; b < table.Length; b++) map[table[b]] = (byte)b;
        return map;
    }
}

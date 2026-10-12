using System.Text.Json;
using System.Text;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

public sealed class StoriesTokenizerTests
{
    private static string Fixture => Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/tokenizer.gguf");

    [Fact]
    public void OwnedVocabularyFitsBudgetAndRejectsChangedScores()
    {
        using var file = new GgufFile(Fixture);
        _ = new StoriesTokenizer(file);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var tokenizer = new StoriesTokenizer(file);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated <= 16*1024*1024,$"Vocabulary construction allocated {allocated} bytes.");
        Assert.Equal(new int[] {1},tokenizer.Encode(""));
        long offset = file.Metadata["tokenizer.ggml.scores"].Offset + 12;
        byte[] bytes = File.ReadAllBytes(Fixture);
        bytes[checked((int)offset)] ^= 1;
        using var changed = new MappedFileFixture(bytes.Length,0,bytes);
        using var invalid = new GgufFile(changed.Path);
        Assert.Throws<NotSupportedException>(()=>new StoriesTokenizer(invalid));
    }

    [Fact]
    public void EncodingPiecesAndWholeDecodingMatchAllOfficialCases()
    {
        using var file = new GgufFile(Fixture);
        var tokenizer = new StoriesTokenizer(file);
        file.Dispose();
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/official-f32/reference.json")));
        foreach (var item in reference.RootElement.GetProperty("tokenizer_cases").EnumerateArray())
        {
            string text = item.GetProperty("text").GetString()!;
            int[] expected = item.GetProperty("tokens").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            var actual = tokenizer.Encode(text,item.GetProperty("add_special").GetBoolean());
            Assert.True(expected.SequenceEqual(actual), $"{item.GetProperty("id")}: expected [{string.Join(',',expected)}], actual [{string.Join(',',actual)}]");
            string[] pieces = item.GetProperty("piece_hex").EnumerateArray().Select(v=>v.GetString()!).ToArray();
            for (int i = 0; i < actual.Length; i++) Assert.Equal(pieces[i],Convert.ToHexString(tokenizer.Piece(actual[i])).ToLowerInvariant());
            Assert.Equal(item.GetProperty("decoded_utf8_hex").GetString(),Convert.ToHexString(tokenizer.DecodeBytes(actual)).ToLowerInvariant());
        }
    }

    [Fact]
    public void StreamingCombinesByteFallbackAndPreservesContinuationSpaces()
    {
        using var file = new GgufFile(Fixture);
        var tokenizer = new StoriesTokenizer(file);
        var decoder = tokenizer.CreateDecoder();
        Assert.Equal("",decoder.Append(3+0xF0));
        Assert.Equal("",decoder.Append(3+0x9F));
        Assert.Equal("",decoder.Append(3+0x98));
        Assert.Equal("😀",decoder.Append(3+0x80));
        Assert.Equal("",decoder.Flush());
        decoder = tokenizer.CreateDecoder();
        Assert.Equal("Once",decoder.Append(9038));
        Assert.Equal(" upon",decoder.Append(2501));
        Assert.Equal("",decoder.Append(StoriesTokenizer.Eos));
        Assert.Equal("",decoder.Flush());
        decoder = tokenizer.CreateDecoder(2501);
        Assert.Equal(" Once",decoder.Append(9038));
    }

    [Fact]
    public void StrictInputBoundsInvalidIdsAndIncompleteBytesHaveDefinedResults()
    {
        using var file = new GgufFile(Fixture);
        var tokenizer = new StoriesTokenizer(file);
        Assert.Throws<EncoderFallbackException>(()=>tokenizer.Encode("\uD800"));
        Assert.Throws<ArgumentOutOfRangeException>(()=>tokenizer.Encode(new string('é',8193)));
        Assert.NotEmpty(tokenizer.Encode(new string('a',16384)));
        Assert.Throws<ArgumentOutOfRangeException>(()=>tokenizer.Decode([32000]));
        Assert.Throws<ArgumentOutOfRangeException>(()=>tokenizer.Decode([-1]));
        var decoder = tokenizer.CreateDecoder();
        Assert.Equal("",decoder.Append(3+0xE2));
        Assert.Equal("�",decoder.Flush());
    }
}

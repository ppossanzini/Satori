using System.Text.Json;
using Satori.Core.Tokenization;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

/// <summary>
/// DS06: byte-level BPE identity against the official tokenizer on Italian, C#, Unicode, whitespace and
/// special tokens, plus decoding and the ChatML template.
/// </summary>
public sealed class BpeTokenizerTests
{
    private static readonly string Fixture =
        Path.Combine(GgufTests.Root, "fixtures/tokenizers/qwen2.5-coder-1.5b");

    private static BpeTokenizer Open() => GgufBpeTokenizer.Open(Path.Combine(Fixture, "tokenizer.gguf"));

    private static JsonElement Reference() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixture, "reference.json"))).RootElement;

    [Fact]
    public void VocabularySizeAndSpecialTokensMatchReference()
    {
        var reference = Reference();
        var tokenization = GgufBpeTokenizer.Read(Path.Combine(Fixture, "tokenizer.gguf"));
        Assert.Equal(reference.GetProperty("vocabulary_size").GetInt32(), tokenization.Tokens.Count);
        Assert.Equal(reference.GetProperty("merge_count").GetInt32(), tokenization.Merges.Count);
        Assert.Equal(reference.GetProperty("bos_token_id").GetInt32(), tokenization.BosTokenId);
        Assert.Equal(reference.GetProperty("eos_token_id").GetInt32(), tokenization.EosTokenId);
        Assert.Contains(tokenization.AddedTokens, t => t.Content == "<|im_start|>" && t.Special);
    }

    [Fact]
    public void EncodingMatchesOfficialSequences()
    {
        var tokenizer = Open();
        var reference = Reference();
        foreach (var testCase in reference.GetProperty("cases").EnumerateArray())
        {
            string text = testCase.GetProperty("text").GetString()!;
            var expected = testCase.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var actual = tokenizer.Encode(text);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void DecodingRoundTrips()
    {
        var tokenizer = Open();
        foreach (var testCase in Reference().GetProperty("cases").EnumerateArray())
        {
            string text = testCase.GetProperty("text").GetString()!;
            var ids = tokenizer.Encode(text);
            Assert.Equal(text, tokenizer.Decode(ids, skipSpecialTokens: false));
        }
    }

    [Fact]
    public void SpecialTokensAreRecognizedAndSkippedOnDecode()
    {
        var tokenizer = Open();
        var ids = tokenizer.Encode("<|im_start|>user\nciao<|im_end|>");
        Assert.Equal(151644, ids[0]);
        Assert.Contains(151645, ids);
        Assert.True(tokenizer.IsSpecial(151644));
        Assert.Equal("user\nciao", tokenizer.Decode(ids));
        Assert.Equal("<|im_start|>user\nciao<|im_end|>", tokenizer.Decode(ids, skipSpecialTokens: false));
    }

    [Fact]
    public void ChatMlTemplateMatchesOfficialRender()
    {
        var reference = Reference();
        var chat = reference.GetProperty("chat_template_case");
        var messages = chat.GetProperty("messages").EnumerateArray()
            .Select(m => (m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!))
            .ToArray();
        bool addPrompt = chat.GetProperty("add_generation_prompt").GetBoolean();
        string expected = chat.GetProperty("text").GetString()!;

        var template = new ChatMlTemplate("<|im_start|>", "<|im_end|>");
        Assert.Equal(expected, template.Render(messages, addPrompt));
    }
}

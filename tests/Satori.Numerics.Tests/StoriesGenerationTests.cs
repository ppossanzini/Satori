using System.Text;
using System.Text.Json;
using Satori.Formats.Gguf;

namespace Satori.Numerics.Tests;

[Trait("Category", "ModelIntegration")]
public sealed class StoriesGenerationTests
{
    private static string ModelPath => Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf");

    [Theory]
    [InlineData("empty",false,"Once upon a time, there was a little girl named L")]
    [InlineData("story",false,", there was a little girl named Lily. She loved")]
    [InlineData("italian",false," The sun was shining and the birds were singing. C")]
    [InlineData("empty",true,"Once upon a time, there was a little girl named L")]
    [InlineData("story",true,", there was a little girl named Lily. She loved")]
    [InlineData("italian",true," The sun was shining and the birds were singing. C")]
    public void TwelveGreedyTokensMatchOfficialSequencesAndStreaming(string id, bool scalar, string expectedText)
    {
        using var fixture = JsonDocument.Parse(File.ReadAllText(Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/official-f32/reference.json")));
        var root = fixture.RootElement;
        var forward = root.GetProperty("forward_cases").EnumerateArray().Single(c=>c.GetProperty("id").GetString()==id);
        string prompt = root.GetProperty("tokenizer_cases").EnumerateArray().First(c=>c.GetProperty("id").GetString()==id).GetProperty("text").GetString()!;
        int[] expected = forward.GetProperty("greedy_tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        using var generator = new StoriesGenerator(ModelPath);
        var streamed = new StringBuilder();
        var result = generator.Generate(prompt,new(12,scalar),part=>streamed.Append(part));
        Assert.Equal(expected,result.Tokens);
        Assert.Equal(expectedText,result.Text);
        Assert.Equal(result.Text,streamed.ToString());
        Assert.Equal(forward.GetProperty("tokens").GetArrayLength(),result.PromptTokenCount);
        Assert.Equal(StoriesStopReason.TokenLimit,result.StopReason);
    }

    [Fact]
    public void SamplingResetsSeedPerGenerationAndValidatesBeforeZeroBudget()
    {
        using var generator = new StoriesGenerator(ModelPath);
        var options = new StoriesGenerationOptions(12, Sampling:new(0.8,40,0.9,123));
        var streamed = new StringBuilder();
        var first = generator.Generate("Once upon a time",options,part=>streamed.Append(part));
        var second = generator.Generate("Once upon a time",options);
        Assert.Equal(first.Tokens,second.Tokens);
        Assert.Equal(first.Text,streamed.ToString());
        Assert.Equal(first.Tokens,generator.Generate("Once upon a time",options with {Scalar=true}).Tokens);
        Assert.Throws<ArgumentOutOfRangeException>(()=>generator.Generate("",new(0,Sampling:new(TopP:0))));
    }

    [Fact]
    public void ZeroFullAndOversizedContextsRespectLimitsWithoutOutput()
    {
        using var generator = new StoriesGenerator(ModelPath);
        using var file = new GgufFile(ModelPath);
        var tokenizer = new StoriesTokenizer(file);
        int emitted = 0;
        var zero = generator.Generate("",new(0),_=>emitted++);
        Assert.Empty(zero.Tokens); Assert.Equal("",zero.Text);
        Assert.Equal(StoriesStopReason.TokenLimit,zero.StopReason);
        string full = string.Join(' ',Enumerable.Repeat("a",127));
        Assert.Equal(128,tokenizer.Encode(full).Length);
        var result = generator.Generate(full,new(int.MaxValue),_=>emitted++);
        Assert.Empty(result.Tokens); Assert.Equal("",result.Text);
        Assert.Equal(StoriesStopReason.ContextLimit,result.StopReason);
        Assert.Throws<ArgumentException>(()=>generator.Generate(full+" a"));
        Assert.Throws<ArgumentOutOfRangeException>(()=>generator.Generate("",new(-1)));
        Assert.Equal(0,emitted);
        string almost = string.Join(' ',Enumerable.Repeat("a",126));
        var last = generator.Generate(almost,new(10));
        Assert.Equal(127,last.PromptTokenCount);
        Assert.Single(last.Tokens);
        Assert.Equal(StoriesStopReason.ContextLimit,last.StopReason);
    }

    [Fact]
    public void DefaultBudgetAndRepeatedCallsAreDeterministic()
    {
        using var generator = new StoriesGenerator(ModelPath);
        var first = generator.Generate("");
        var second = generator.Generate("");
        Assert.Equal(32,first.Tokens.Count);
        Assert.Equal(StoriesStopReason.TokenLimit,first.StopReason);
        Assert.Equal(first.Tokens,second.Tokens);
        Assert.Equal(first.Text,second.Text);
    }

    [Fact]
    public void CallbackFailureReleasesSessionAndGeneratorLifetimeIsExplicit()
    {
        using var generator = new StoriesGenerator(ModelPath);
        Assert.Throws<InvalidOperationException>(()=>generator.Generate("",new(12),_=>throw new InvalidOperationException()));
        var result = generator.Generate("",new(1));
        Assert.Equal(new int[] {9038},result.Tokens);
        Assert.Equal("Once",result.Text);
        generator.Dispose();
        generator.Dispose();
        Assert.Throws<ObjectDisposedException>(()=>generator.Generate(""));
        Assert.Throws<NotSupportedException>(()=>new StoriesGenerator(Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/tokenizer.gguf")));
    }
}

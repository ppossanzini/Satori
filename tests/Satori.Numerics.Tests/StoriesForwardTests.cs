using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using Satori.Formats.Gguf;
using Xunit.Abstractions;

namespace Satori.Numerics.Tests;

[Trait("Category", "ModelIntegration")]
public sealed class StoriesForwardTests(ITestOutputHelper output)
{
    private static string ModelPath => Path.Combine(GgufTests.Root,".temp/gguf-corpus/stories15M-q8_0.gguf");
    private static string References => Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/numpy");

    [Theory]
    [InlineData("empty",false)]
    [InlineData("story",false)]
    [InlineData("italian",false)]
    [InlineData("empty",true)]
    [InlineData("story",true)]
    [InlineData("italian",true)]
    public void EveryStateAndLogitMatchesIndependentExactWeightOracle(string id, bool scalar)
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(References,"reference.json")));
        var item = reference.RootElement.GetProperty("forward_cases").EnumerateArray().Single(v=>v.GetProperty("id").GetString()==id);
        int[] tokens = item.GetProperty("tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        var states = item.GetProperty("intermediates").EnumerateArray().ToDictionary(v=>v.GetProperty("name").GetString()!,v=>v);
        var observed = new HashSet<string>();
        double maxError = 0, maxRms = 0;
        void Compare(JsonElement record, ReadOnlySpan<float> actual)
        {
            byte[] bytes = File.ReadAllBytes(Path.Combine(References,record.GetProperty("file").GetString()!));
            Assert.Equal(record.GetProperty("sha256").GetString(),Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant());
            Assert.Equal(record.GetProperty("elements").GetInt32(),actual.Length);
            double squared = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                double expected = BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(i*8,8));
                double error = Math.Abs(actual[i]-expected);
                Assert.True(float.IsFinite(actual[i]) && error <= 5e-4 + 5e-5*Math.Abs(expected),$"{record.GetProperty("file")}[{i}]: {actual[i]} vs {expected}");
                maxError = Math.Max(maxError,error); squared += error*error;
            }
            double rms = Math.Sqrt(squared/actual.Length);
            Assert.True(rms<=1e-4,$"RMS {rms}"); maxRms = Math.Max(maxRms,rms);
        }
        using var model = new StoriesModel(ModelPath);
        float[] logits = new float[StoriesModel.VocabularySize];
        model.Forward(tokens,logits,scalar,(name,values)=>
        {
            Assert.True(observed.Add(name),$"Duplicate state {name}");
            Compare(states[name],values);
        });
        Assert.Equal(15,observed.Count);
        Compare(item.GetProperty("logits"),logits);
        int best = Enumerable.Range(0,logits.Length).MaxBy(i=>logits[i]);
        Assert.Equal(item.GetProperty("greedy_tokens")[0].GetInt32(),best);
        output.WriteLine($"{id}, scalar={scalar}: max error {maxError:E6}; max RMS {maxRms:E6}; argmax {best}");
    }

    [Fact]
    public void CallsRetainNoSequenceStateAndRejectInvalidInputsBeforeWriting()
    {
        using var model = new StoriesModel(ModelPath);
        float[] first = new float[32000], repeated = new float[32000], sentinel = Enumerable.Repeat(123f,32000).ToArray();
        model.Forward([1],first);
        model.Forward([1,9038,2501],repeated);
        model.Forward([1],repeated);
        Assert.Equal(first,repeated);
        Assert.Throws<ArgumentOutOfRangeException>(()=>model.Forward([],sentinel));
        Assert.Throws<ArgumentOutOfRangeException>(()=>model.Forward(new int[129],sentinel));
        Assert.Throws<ArgumentOutOfRangeException>(()=>model.Forward([1,-1],sentinel));
        Assert.Throws<ArgumentOutOfRangeException>(()=>model.Forward([32000],sentinel));
        Assert.Throws<ArgumentException>(()=>model.Forward([1],new float[31999]));
        Assert.All(sentinel,value=>Assert.Equal(123f,value));
        Assert.Throws<InvalidOperationException>(()=>model.Forward([1],repeated,observer:(_,_)=>throw new InvalidOperationException()));
        model.Forward([1],repeated);
        Assert.Equal(first,repeated);
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(()=>model.Forward([1],repeated));
    }

    [Fact]
    public void FullContextUsesBoundedTemporaryMemoryAndHasFiniteLogits()
    {
        using var model = new StoriesModel(ModelPath);
        float[] logits = new float[32000];
        model.Forward([1],logits);
        int[] tokens = Enumerable.Repeat(9038,128).ToArray(); tokens[0] = 1;
        long before = GC.GetAllocatedBytesForCurrentThread();
        model.Forward(tokens,logits);
        long allocated = GC.GetAllocatedBytesForCurrentThread()-before;
        Assert.True(allocated<=2*1024*1024,$"Forward allocated {allocated} bytes.");
        Assert.All(logits,value=>Assert.True(float.IsFinite(value)));
        output.WriteLine($"128-token temporary allocation: {allocated} bytes; logits caller-owned.");
    }

    [Fact]
    public void DifferentArtifactIsRejected()
    {
        Assert.Throws<NotSupportedException>(()=>new StoriesModel(Path.Combine(GgufTests.Root,"fixtures/models/stories15m-q8-v1/tokenizer.gguf")));
        byte[] changed = File.ReadAllBytes(ModelPath); changed[^1] ^= 1;
        using var fixture = new MappedFileFixture(changed.Length,0,changed);
        Assert.Throws<NotSupportedException>(()=>new StoriesModel(fixture.Path));
    }
}

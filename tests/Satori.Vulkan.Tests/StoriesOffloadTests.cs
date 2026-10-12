using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Formats.Gguf;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class StoriesOffloadTests
{
    private static string Root => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"../../../../../"));
    private static string Model => Path.Combine(Root,".temp/gguf-corpus/stories15M-q8_0.gguf");

    [Theory]
    [InlineData(StoriesVulkanPath.MatVec)]
    [InlineData(StoriesVulkanPath.Resident)]
    public void OffloadedCacheReachesContextLimitAndGeneratorBorrowsModel(StoriesVulkanPath path)
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        using var runtime=new VulkanStoriesRuntime(Model,new(StoriesBackendMode.VulkanRequired,new(Validation:true),Path:path));
        using var cpu=new StoriesModel(Model);
        using var gpuSession=runtime.Model.CreateSession();
        using var cpuSession=cpu.CreateSession();
        float[] actual=new float[32000],expected=new float[32000];
        for(int position=0;position<128;position++)
        {
            int token=position==0?1:position%2==0?9038:2501;
            Assert.True(gpuSession.TryAppend(token,actual));
            Assert.True(cpuSession.TryAppend(token,expected));
            double squared=0;
            for(int i=0;i<actual.Length;i++)
            {
                double error=Math.Abs(actual[i]-expected[i]);
                Assert.True(float.IsFinite(actual[i]) && error<=5e-4+5e-5*Math.Abs(expected[i]));
                squared+=error*error;
            }
            Assert.True(Math.Sqrt(squared/actual.Length)<=1e-4);
        }
        if(path==StoriesVulkanPath.MatVec)Assert.Equal(128*43,runtime.Statistics!.CompletedDispatches);
        else {Assert.Equal(128,runtime.ResidentStatistics!.SubmissionCount);Assert.Equal(128*129152,runtime.ResidentStatistics.TransferBytes);Assert.Equal(128*43,runtime.ResidentStatistics.MatVecDispatches);}
        float[] saved=actual.ToArray();
        Assert.False(gpuSession.TryAppend(1,actual));Assert.Equal(saved,actual);
        gpuSession.Reset();Assert.True(gpuSession.TryAppend(1,actual));
        using(var generator=new StoriesGenerator(Model,runtime.Model))
        {
            var text=new System.Text.StringBuilder();
            var result=generator.Generate("",new(12),part=>text.Append(part));
            Assert.Equal("Once upon a time, there was a little girl named L",result.Text);
            Assert.Equal(result.Text,text.ToString());
        }
        Assert.True(gpuSession.TryAppend(9038,actual));
        runtime.Dispose();Assert.Equal(0,runtime.ValidationErrorCount);
    }

    [Theory]
    [InlineData(StoriesVulkanPath.MatVec)]
    [InlineData(StoriesVulkanPath.Resident)]
    public void OffloadedForwardAndIncrementalLogitsMatchIndependentOracleAndGreedySequences(StoriesVulkanPath path)
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        using var runtime=new VulkanStoriesRuntime(Model,new(StoriesBackendMode.VulkanRequired,new(Validation:true),Path:path));
        Assert.False(runtime.Device!.Software);
        Assert.Equal(43,path==StoriesVulkanPath.MatVec?runtime.Statistics!.ResidentMatrices:runtime.ResidentStatistics!.Matrices);
        string directory=Path.Combine(Root,"fixtures/models/stories15m-q8-v1/numpy");
        using var reference=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"reference.json")));
        foreach(var item in reference.RootElement.GetProperty("forward_cases").EnumerateArray())
        {
            int[] tokens=item.GetProperty("tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
            float[] logits=new float[32000],incremental=new float[32000];
            var states=item.GetProperty("intermediates").EnumerateArray().ToDictionary(v=>v.GetProperty("name").GetString()!,v=>v);
            int seen=0;
            runtime.Model.Forward(tokens,logits,observer:(name,values)=>{Compare(states[name],values);seen++;});
            Assert.Equal(15,seen);
            Compare(item.GetProperty("logits"),logits);
            using var session=runtime.Model.CreateSession();
            foreach(int token in tokens) Assert.True(session.TryAppend(token,incremental));
            Compare(item.GetProperty("logits"),incremental);
            Assert.Equal(tokens.Length,session.Count);
            session.Reset();
            foreach(int token in tokens) Assert.True(session.TryAppend(token,incremental));
            Compare(item.GetProperty("logits"),incremental);
            foreach(int token in item.GetProperty("greedy_tokens").EnumerateArray().Select(v=>v.GetInt32()))
            {
                Assert.Equal(token,Enumerable.Range(0,32000).MaxBy(i=>incremental[i]));
                Assert.True(session.TryAppend(token,incremental));
            }
        }
        long before=path==StoriesVulkanPath.MatVec?runtime.Statistics!.CompletedDispatches:runtime.ResidentStatistics!.MatVecDispatches;
        runtime.Model.Forward([1],new float[32000],scalar:true);
        Assert.Equal(before,path==StoriesVulkanPath.MatVec?runtime.Statistics!.CompletedDispatches:runtime.ResidentStatistics!.MatVecDispatches);
        runtime.Dispose();
        Assert.Equal(0,runtime.ValidationErrorCount);

        void Compare(JsonElement record,ReadOnlySpan<float> actual)
        {
            byte[] bytes=File.ReadAllBytes(Path.Combine(directory,record.GetProperty("file").GetString()!));
            Assert.Equal(record.GetProperty("sha256").GetString(),Convert.ToHexStringLower(SHA256.HashData(bytes)));
            Assert.Equal(record.GetProperty("elements").GetInt32(),actual.Length);
            double squared=0;
            for(int i=0;i<actual.Length;i++)
            {
                double expected=BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(i*8,8));
                double error=Math.Abs(actual[i]-expected);
                Assert.True(float.IsFinite(actual[i]) && error<=5e-4+5e-5*Math.Abs(expected));
                squared+=error*error;
            }
            Assert.True(Math.Sqrt(squared/actual.Length)<=1e-4);
        }
    }
}

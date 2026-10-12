using System.Buffers.Binary;
using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Formats.Gguf;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanResidentLayerTests
{
    [Fact]
    public void WholeResidentLayersMatchIndependentBoundariesWithOneSubmission()
    {
        string root=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"../../../../../")),path=Path.Combine(root,".temp/gguf-corpus/stories15M-q8_0.gguf"),directory=Path.Combine(root,"fixtures/models/stories15m-q8-v1/numpy");
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        Assert.True(VulkanContext.TryCreate(new(Validation:true),out var created,out var availability),availability.ToString());
        using var context=created!;using var model=new VulkanResidentModel(path,context);using var file=new GgufFile(path);var embedding=new GgufQuantizedMatrix(file,"token_embd.weight");
        using var reference=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"reference.json")));
        foreach(var item in reference.RootElement.GetProperty("forward_cases").EnumerateArray())
        {
            using var layer=model.CreateLayerSession(0);float[] input=new float[288],output=new float[288];int count=0;
            foreach(int token in item.GetProperty("tokens").EnumerateArray().Select(x=>x.GetInt32())) {embedding.ReadRow(token,input);Assert.True(layer.TryAppend(input,output));count++;}
            Compare(item.GetProperty("intermediates").EnumerateArray().Single(s=>s.GetProperty("name").GetString()=="l_out-0"),output);
            Assert.Equal(count,layer.Count);layer.Reset();Assert.Equal(0,layer.Count);
        }
        var first=reference.RootElement.GetProperty("forward_cases")[0];float[] state=new float[288];embedding.ReadRow(1,state);
        for(int i=0;i<6;i++)
        {
            using var layer=model.CreateLayerSession(i);float[] output=new float[288];var before=model.Statistics;
            Assert.True(layer.TryAppend(state,output));var after=model.Statistics;
            Assert.Equal(1,after.SubmissionCount-before.SubmissionCount);Assert.Equal(7,after.MatVecDispatches-before.MatVecDispatches);Assert.Equal(2304,after.TransferBytes-before.TransferBytes);
            var record=first.GetProperty("intermediates").EnumerateArray().Single(s=>s.GetProperty("name").GetString()==$"l_out-{i}");Compare(record,output);state=Read(record);
        }
        Assert.Equal(43,model.Statistics.Matrices);Assert.Equal(16137216UL,model.Statistics.WeightBytes);
        model.Dispose();context.Dispose();Assert.Equal(0,context.ValidationErrorCount);
        float[] Read(JsonElement record)
        {
            byte[] bytes=File.ReadAllBytes(Path.Combine(directory,record.GetProperty("file").GetString()!));
            return Enumerable.Range(0,record.GetProperty("elements").GetInt32()).Select(i=>(float)BinaryPrimitives.ReadDoubleLittleEndian(bytes.AsSpan(i*8,8))).ToArray();
        }
        void Compare(JsonElement record,float[] actual)
        {
            var expected=Read(record);double squared=0;
            for(int i=0;i<actual.Length;i++){double error=Math.Abs(actual[i]-expected[i]);Assert.True(float.IsFinite(actual[i])&&error<=5e-4+5e-5*Math.Abs(expected[i]),$"{i}: {actual[i]} vs {expected[i]}");squared+=error*error;}
            Assert.True(Math.Sqrt(squared/actual.Length)<=1e-4);
        }
    }
}

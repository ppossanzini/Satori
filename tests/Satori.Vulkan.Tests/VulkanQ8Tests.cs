using System.Buffers.Binary;
using Satori.Backends.Vulkan;
using Satori.Core;
using Satori.Formats.Gguf;
using Satori.Validation;

namespace Satori.Vulkan.Tests;

[Trait("Category","VulkanHardware")]
public sealed class VulkanQ8Tests
{
    private static VulkanContext Create()
    {
        Assert.Equal("1",Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"));
        Assert.True(VulkanContext.TryCreate(new(Validation:true),out var context,out var availability),availability.ToString());
        Assert.False(context!.Info.Software);
        return context;
    }

    [Fact]
    public void PackedSignedScalesAlignmentAndRowTailsMatchFp64AndCpu()
    {
        using var context=Create();
        foreach(var sample in Q8Validation.CriticalCases())
        {
            using var gpu=context.CreateQ8Matrix(sample.Payload,sample.Shape,new(MeasureKernelTime:true));
            float[] output=new float[sample.Shape.Rows],scalar=new float[output.Length],auto=new float[output.Length];
            QuantizedMatVec.ExecuteScalar(sample.Payload,sample.Shape,sample.Input,scalar);
            QuantizedMatVec.Execute(sample.Payload,sample.Shape,sample.Input,auto);
            var time=gpu.Execute(sample.Input,output);
            Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,sample.Input,output).Passed,sample.Name);
            Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,sample.Input,scalar).Passed);
            Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,sample.Input,auto).Passed);
            if(context.Info.TimestampValidBits>0) Assert.True(time.KernelMilliseconds is >=0 and <10000);
            Assert.True(time.HostMilliseconds>=0);
            for(int repeat=0;repeat<3;repeat++)
            {
                float[] input=sample.Input.Select(v=>v*(repeat+1)).ToArray();
                gpu.Execute(input,output);
                Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,input,output).Passed);
            }
        }
        context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public void OwnedWeightsSurviveCallerMutationAndMappedFileDisposal()
    {
        using var context=Create();
        var sample=Q8Validation.CriticalCases().First();
        using var owned=context.CreateQ8Matrix(sample.Payload,sample.Shape);
        byte[] reference=sample.Payload.ToArray();
        Array.Fill(sample.Payload,(byte)0);
        float[] output=new float[sample.Shape.Rows];
        owned.Execute(sample.Input,output);
        Assert.True(Q8Validation.Compare(reference,sample.Shape,sample.Input,output).Passed);
        string root=Path.GetFullPath(Path.Combine(AppContext.BaseDirectory,"../../../../../"));
        VulkanQ8Matrix? mapped=null;
        using(var file=new GgufFile(Path.Combine(root,"fixtures/gguf/v1/q8_0.gguf")))
            file.ReadTensor("weights",0,34,(bytes,owner)=>mapped=owner.CreateQ8Matrix(bytes,new(1,32,BlockQuantization.Q8_0)),context);
        using(mapped)
        {
            float[] one=new float[1];
            mapped!.Execute(Enumerable.Repeat(1f,32).ToArray(),one);
            Assert.Equal(-8f,one[0]);
        }
        context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public void InvalidPayloadVectorsAndDisposedOwnersNeverCommitOutput()
    {
        using var context=Create();
        foreach(ushort bits in new ushort[] {0x7c00,0xfc00,0x7e00})
        {
            byte[] bad=new byte[34]; BinaryPrimitives.WriteUInt16LittleEndian(bad,bits);
            Assert.Throws<InvalidDataException>(()=>context.CreateQ8Matrix(bad,new(1,32,BlockQuantization.Q8_0)));
        }
        Assert.Throws<NotSupportedException>(()=>context.CreateQ8Matrix(new byte[18],new(1,32,BlockQuantization.Q4_0)));
        Assert.Throws<ArgumentException>(()=>context.CreateQ8Matrix([],new(1,32,BlockQuantization.Q8_0)));
        var shape=new QuantizedMatrixShape(64,64,BlockQuantization.Q8_0);
        using var matrix=context.CreateQ8Matrix(new byte[shape.ByteLength],shape);
        float[] sentinel=Enumerable.Repeat(17f,64).ToArray();
        Assert.Throws<ArgumentException>(()=>matrix.Execute(new float[32],sentinel));
        Assert.Throws<ArgumentException>(()=>matrix.Execute(sentinel,sentinel));
        float[] invalid=new float[64];invalid[17]=float.NaN;
        Assert.Throws<ArgumentException>(()=>matrix.Execute(invalid,sentinel));
        Assert.All(sentinel,v=>Assert.Equal(17f,v));
        context.Dispose(); matrix.Dispose(); matrix.Dispose();
        Assert.Throws<ObjectDisposedException>(()=>matrix.Execute(new float[64],sentinel));
        Assert.Throws<ObjectDisposedException>(()=>context.CreateQ8Matrix(new byte[34],new(1,32,BlockQuantization.Q8_0)));
        Assert.All(sentinel,v=>Assert.Equal(17f,v));
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public void WarmMatVecRetainsWeightBufferWithoutManagedAllocation()
    {
        using var context=Create();
        var sample=Q8Validation.CriticalCases().Last();
        using var matrix=context.CreateQ8Matrix(sample.Payload,sample.Shape,new(true));
        float[] output=new float[sample.Shape.Rows];
        for(int i=0;i<60;i++) matrix.Execute(sample.Input,output);
        ulong allocated=matrix.AllocatedBytes,weights=matrix.WeightAllocationBytes;
        long before=GC.GetAllocatedBytesForCurrentThread();
        for(int i=0;i<10;i++) matrix.Execute(sample.Input,output);
        Assert.Equal(0,GC.GetAllocatedBytesForCurrentThread()-before);
        Assert.Equal(allocated,matrix.AllocatedBytes);
        Assert.Equal(weights,matrix.WeightAllocationBytes);
        matrix.Dispose();
        Assert.Equal(0UL,matrix.AllocatedBytes);
        context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }

    [Fact]
    public async Task MatrixAndContextDisposalSerializeWithConcurrentExecution()
    {
        var context=Create();
        var sample=Q8Validation.CriticalCases().Last();
        var matrix=context.CreateQ8Matrix(sample.Payload,sample.Shape);
        await Task.WhenAll(Enumerable.Range(0,4).Select(_=>Task.Run(()=>
        {
            float[] output=new float[sample.Shape.Rows];
            for(int i=0;i<10;i++) { matrix.Execute(sample.Input,output); Assert.True(Q8Validation.Compare(sample.Payload,sample.Shape,sample.Input,output).Passed); }
        })));
        using var start=new ManualResetEventSlim();
        var work=Task.Run(()=>
        {
            start.Wait();
            try { matrix.Execute(sample.Input,new float[sample.Shape.Rows]); } catch(ObjectDisposedException) { }
        });
        var dispose=Task.Run(()=> {start.Wait();matrix.Dispose();context.Dispose();});
        start.Set(); await Task.WhenAll(work,dispose);
        matrix.Dispose();context.Dispose();
        Assert.Equal(0,context.ValidationErrorCount);
    }
}

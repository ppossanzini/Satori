using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Core;
using Satori.Formats.Gguf;
using Satori.Validation;

internal static class Q8ValidationRunner
{
    internal static int Run(string modelPath)
    {
        if(Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC")!="1")
            throw new InvalidOperationException("Set VK_VALIDATION_VALIDATE_SYNC=1.");
        if(!VulkanContext.TryCreate(new(Validation:true),out var created,out var availability))
            throw new InvalidOperationException(availability.ToString());
        using var context=created!;
        if(context.Info.Software) throw new InvalidOperationException("Hardware GPU required.");
        var cases=new List<object>();
        foreach(var sample in Q8Validation.CriticalCases()) Check(sample.Name,sample.Payload,sample.Shape,sample.Input,false);
        using(var file=new GgufFile(modelPath))
        {
            foreach(var tensor in file.Tensors.Values.Where(t=>t.Type==8 && t.Dimensions.Count==2))
            {
                byte[] payload=new byte[checked((int)tensor.ByteLength)];
                file.ReadTensor(tensor.Name,0,payload.Length,static(bytes,target)=>bytes.CopyTo(target),payload);
                var shape=new QuantizedMatrixShape(checked((int)tensor.Dimensions[1]),checked((int)tensor.Dimensions[0]),BlockQuantization.Q8_0);
                float[] input=Enumerable.Range(0,shape.Columns).Select(i=>(i*17%31-15)/16f).ToArray();
                Check(tensor.Name,payload,shape,input,true);
            }
        }
        context.Dispose();
        if(context.ValidationErrorCount!=0) throw new InvalidOperationException("Vulkan validation failed.");
        var report=new {TimestampUtc=DateTimeOffset.UtcNow,Runtime=RuntimeInformation.FrameworkDescription,
            ModelSha256=Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(modelPath))),Device=context.Info,
            ValidationErrors=context.ValidationErrorCount,ValidationMessages=context.ValidationMessages,
            TimingScope="Kernel: compute-stage timestamps around dispatch. Host: synchronous vector upload, dispatch, fence, query and readback; resident weights excluded. Three warmups, fifteen samples. Sequential measurements.",Cases=cases};
        Directory.CreateDirectory("reports/vulkan");
        File.WriteAllText("reports/vulkan/q8-matvec.json",JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Q8: {cases.Count} workloads passed FP64 bounds; validation errors: {context.ValidationErrorCount}.");
        return 0;

        void Check(string name,byte[] payload,QuantizedMatrixShape shape,float[] input,bool measure)
        {
            long start=Stopwatch.GetTimestamp();
            using var matrix=context.CreateQ8Matrix(payload,shape,new(true));
            double creationMs=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            float[] output=new float[shape.Rows],scalar=new float[shape.Rows],auto=new float[shape.Rows];
            matrix.Execute(input,output);
            QuantizedMatVec.ExecuteScalar(payload,shape,input,scalar);
            QuantizedMatVec.Execute(payload,shape,input,auto);
            var gpuError=Q8Validation.Compare(payload,shape,input,output);
            var scalarError=Q8Validation.Compare(payload,shape,input,scalar);
            var autoError=Q8Validation.Compare(payload,shape,input,auto);
            if(!gpuError.Passed || !scalarError.Passed || !autoError.Passed) throw new InvalidDataException($"FP64 bound failed: {name}");
            var host=new List<double>();var kernel=new List<double?>();var cpuScalar=new List<double>();var cpuAuto=new List<double>();var allocations=new List<long>();
            if(measure)
            {
                for(int i=0;i<3;i++) {matrix.Execute(input,output);QuantizedMatVec.ExecuteScalar(payload,shape,input,scalar);QuantizedMatVec.Execute(payload,shape,input,auto);}
                for(int i=0;i<15;i++)
                {
                    long before=GC.GetAllocatedBytesForCurrentThread();
                    var timing=matrix.Execute(input,output);
                    long bytes=GC.GetAllocatedBytesForCurrentThread()-before;
                    host.Add(timing.HostMilliseconds);kernel.Add(timing.KernelMilliseconds);allocations.Add(bytes);
                    start=Stopwatch.GetTimestamp();QuantizedMatVec.ExecuteScalar(payload,shape,input,scalar);cpuScalar.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                    start=Stopwatch.GetTimestamp();QuantizedMatVec.Execute(payload,shape,input,auto);cpuAuto.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                }
            }
            cases.Add(new {Name=name,shape.Rows,shape.Columns,RealModel=measure,PayloadBytes=payload.Length,CreationAndWeightUploadMs=creationMs,
                matrix.WeightAllocationBytes,matrix.AllocatedBytes,TransferBytesPerCall=(shape.Rows+shape.Columns)*4,
                GpuError=gpuError,ScalarError=scalarError,AutoError=autoError,HostSamplesMs=host,KernelSamplesMs=kernel,CpuScalarSamplesMs=cpuScalar,CpuAutoSamplesMs=cpuAuto,ManagedAllocatedBytes=allocations});
        }
    }
}

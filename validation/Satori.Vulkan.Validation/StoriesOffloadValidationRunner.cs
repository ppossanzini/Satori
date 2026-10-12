using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Formats.Gguf;

internal static class StoriesOffloadValidationRunner
{
    internal static int Run(string path,StoriesOffloadValidationOptions options)
    {
        if(options.Validation && Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC")!="1")
            throw new InvalidOperationException("Set VK_VALIDATION_VALIDATE_SYNC=1.");
        var reports=new List<object>();
        float[][]? cpuLogits=null;
        foreach(var mode in new[] {StoriesBackendMode.Cpu,StoriesBackendMode.VulkanRequired})
        {
            long start=Stopwatch.GetTimestamp();
            using var runtime=new VulkanStoriesRuntime(path,new(mode,new(Validation:options.Validation)));
            double coldMs=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            if(runtime.Device?.Software==true) throw new InvalidOperationException("Hardware GPU required.");
            float[] logits=new float[32000];
            using var session=runtime.Model.CreateSession();
            int[] prompt=[1,9038,2501];
            var samples=new List<object>();
            var comparison=new List<object>();
            var traces=new List<float[]>();
            for(int i=0;i<prompt.Length+options.DecodeTokens;i++)
            {
                session.TryAppend(i<prompt.Length?prompt[i]:9038,logits);
                if(mode==StoriesBackendMode.Cpu) traces.Add(logits.ToArray());
                else
                {
                    double max=0,squared=0;
                    for(int j=0;j<logits.Length;j++)
                    {
                        double expected=cpuLogits![i][j],error=Math.Abs(logits[j]-expected);
                        if(!float.IsFinite(logits[j]) || error>5e-4+5e-5*Math.Abs(expected)) throw new InvalidDataException("CPU/GPU logits differ.");
                        max=Math.Max(max,error);squared+=error*error;
                    }
                    double rms=Math.Sqrt(squared/logits.Length);
                    if(rms>1e-4) throw new InvalidDataException("CPU/GPU RMS gate failed.");
                    comparison.Add(new {Position=i,MaxAbsoluteError=max,RmsError=rms});
                }
            }
            if(mode==StoriesBackendMode.Cpu) cpuLogits=traces.ToArray();
            for(int iteration=0;iteration<options.Warmups+options.Samples;iteration++)
            {
                session.Reset();
                double[] decode=new double[options.DecodeTokens];
                long before=GC.GetAllocatedBytesForCurrentThread();
                long totalStart=Stopwatch.GetTimestamp();
                foreach(int token in prompt) session.TryAppend(token,logits);
                double prefill=Stopwatch.GetElapsedTime(totalStart).TotalMilliseconds;
                for(int i=0;i<decode.Length;i++)
                {
                    start=Stopwatch.GetTimestamp();session.TryAppend(9038,logits);decode[i]=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                }
                double total=Stopwatch.GetElapsedTime(totalStart).TotalMilliseconds;
                long bytes=GC.GetAllocatedBytesForCurrentThread()-before;
                if(iteration>=options.Warmups) samples.Add(new {PrefillMs=prefill,DecodeSamplesMs=decode,TotalMs=total,ManagedAllocatedBytes=bytes});
            }
            var statistics=runtime.Statistics;
            runtime.Dispose();
            if(runtime.ValidationErrorCount!=0) throw new InvalidOperationException("Vulkan validation failed.");
            reports.Add(new {Backend=mode==StoriesBackendMode.Cpu?"CPU":"Vulkan",ColdInitializationAndWeightsMs=coldMs,
                runtime.Device,Statistics=statistics,CacheBytes=StoriesSession.CacheByteLength,Samples=samples,Comparison=comparison,
                ValidationErrors=runtime.ValidationErrorCount,ValidationMessages=runtime.ValidationMessages});
        }
        var report=new {TimestampUtc=DateTimeOffset.UtcNow,Runtime=RuntimeInformation.FrameworkDescription,Options=options,
            ModelSha256="2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6",
            Scope="Synchronous session prefill [1,9038,2501], then eight teacher-forced token 9038 appends; no tokenization, sampling or text decoding. Reset and array allocation excluded. Statistics include correctness, warmups and measured calls; kernel excludes CPU operators and transfers. Host total includes all CPU operators and GPU vector transfers. Sequential CPU then GPU.",Reports=reports};
        Directory.CreateDirectory("reports/vulkan");
        string name=options.Validation?"offload-validation.json":"offload-benchmark.json";
        File.WriteAllText(Path.Combine("reports/vulkan",name),JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true}));
        Console.WriteLine($"Offload verified; report: reports/vulkan/{name}");
        return 0;
    }
}

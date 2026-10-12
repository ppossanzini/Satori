using System.Diagnostics;
using System.Text.Json;
using Satori.Backends.Vulkan;

internal static class StoriesPathContextRunner
{
    internal static int Run(string path)
    {
        var cases=new List<object>();
        foreach(string name in new[] {"CPU","MatVec","Resident"})
        {
            long start=Stopwatch.GetTimestamp();
            using var runtime=new VulkanStoriesRuntime(path,new(name=="CPU"?StoriesBackendMode.Cpu:StoriesBackendMode.VulkanRequired,
                Path:name=="Resident"?StoriesVulkanPath.Resident:StoriesVulkanPath.MatVec,MatVec:name=="MatVec"?new():null));
            double startup=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            using var session=runtime.Model.CreateSession();float[] logits=new float[32000];
            var contexts=new List<object>();
            foreach(int prefix in new[] {1,64})
            {
                session.Reset();for(int i=0;i<prefix;i++)session.TryAppend(i==0?1:9038,logits);
                var samples=new List<double>();var allocations=new List<long>();
                for(int i=0;i<10;i++)
                {
                    long before=GC.GetAllocatedBytesForCurrentThread();start=Stopwatch.GetTimestamp();session.TryAppend(9038,logits);
                    double elapsed=Stopwatch.GetElapsedTime(start).TotalMilliseconds;long bytes=GC.GetAllocatedBytesForCurrentThread()-before;
                    if(i>=3){samples.Add(elapsed);allocations.Add(bytes);}
                }
                contexts.Add(new {PrefixBeforeWarmup=prefix,MeasuredPositions=Enumerable.Range(prefix+3,7).ToArray(),SamplesMs=samples,ManagedAllocatedBytes=allocations});
            }
            cases.Add(new {Path=name,StartupMs=startup,Device=runtime.Device,Contexts=contexts,LegacyStatistics=runtime.Statistics,ResidentStatistics=runtime.ResidentStatistics});
        }
        Directory.CreateDirectory("reports/vulkan");
        File.WriteAllText("reports/vulkan/path-contexts.json",JsonSerializer.Serialize(new {Runtime=System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            Scope="One context prefill excluded, three warmup appends then seven measured sequential appends. Teacher-forced 9038; positions differ within the declared range. CPU/MatVec/Resident sequential; validation and timestamps disabled.",Cases=cases},new JsonSerializerOptions {WriteIndented=true}));
        Console.WriteLine("Short/long context paths measured.");return 0;
    }
}

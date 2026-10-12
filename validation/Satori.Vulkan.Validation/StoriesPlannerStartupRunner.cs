using System.Diagnostics;
using System.Text.Json;
using Satori.Backends.Vulkan;
using Satori.Formats.Gguf;

internal static class StoriesPlannerStartupRunner
{
    internal static int Run(string path,string cachePath)
    {
        long start=Stopwatch.GetTimestamp();
        using var runtime=new VulkanStoriesRuntime(path,new(StoriesBackendMode.Auto,Planner:new(cachePath)));
        double initializationMs=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        using var generator=new StoriesGenerator(path,runtime.Model);
        var result=generator.Generate("",new(1));
        Console.WriteLine(JsonSerializer.Serialize(new {runtime.ActualBackend,runtime.PlanFromCache,runtime.CandidateStatistics,runtime.CandidateDevice,Runtime=System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription,
            InitializationMs=initializationMs,runtime.Plan,Text=result.Text,Tokens=result.Tokens}));
        return 0;
    }
}

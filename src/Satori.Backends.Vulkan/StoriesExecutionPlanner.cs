using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Satori.Formats.Gguf;

namespace Satori.Backends.Vulkan;

internal static class StoriesExecutionPlanner
{
    internal static string GetIdentity(VulkanContext context)
    {
        string cpu=File.Exists("/proc/cpuinfo")?string.Join('\n',File.ReadLines("/proc/cpuinfo").Where(l=>l.StartsWith("model name",StringComparison.Ordinal)||l.StartsWith("flags",StringComparison.Ordinal)).Distinct()):Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")??RuntimeInformation.ProcessArchitecture.ToString();
        string binaries=string.Join(':',new[] {typeof(StoriesModel).Assembly.Location,typeof(Satori.Core.QuantizedMatVec).Assembly.Location,typeof(StoriesExecutionPlanner).Assembly.Location}.Select(p=>Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(p)))));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes($"session-two-tokens-v2|resident-graph-fp32-tree64-small-row64-output-arena477344-v1|{cpu}|{RuntimeInformation.OSDescription}|{RuntimeInformation.FrameworkDescription}|{System.Runtime.Intrinsics.X86.Avx2.IsSupported}|{JsonSerializer.Serialize(context.Info)}|{binaries}")));
    }

    internal static StoriesExecutionPlan? ReadCache(string identity,StoriesPlannerOptions options)
    {
        if(File.Exists(options.CachePath))
        {
            try
            {
                var saved=JsonSerializer.Deserialize<StoriesExecutionPlan>(File.ReadAllText(options.CachePath));
                if(saved?.IsValidFor(identity)==true && saved.ResidentSamplesMs is not null) return saved;
            }
            catch(JsonException) { }
        }
        return null;
    }

    internal static StoriesExecutionPlan Select(string path,StoriesModel gpu,StoriesModel resident,string identity,StoriesPlannerOptions options)
    {
        using var reference=new StoriesModel(path);
        float[] expected=new float[32000],actual=new float[32000];
        reference.Forward([1,9038],expected);
        foreach(var candidate in new[] {gpu,resident})
        {
            candidate.Forward([1,9038],actual);double squared=0;
            for(int i=0;i<actual.Length;i++)
            {
                double error=Math.Abs(actual[i]-expected[i]);
                if(!float.IsFinite(actual[i]) || error>5e-4+5e-5*Math.Abs(expected[i]))throw new InvalidDataException("Planner CPU/GPU numerical gate failed.");
                squared+=error*error;
            }
            if(Math.Sqrt(squared/actual.Length)>1e-4)throw new InvalidDataException("Planner CPU/GPU RMS gate failed.");
        }
        var plan=new StoriesExecutionPlan(identity,Measure(reference),Measure(gpu),Measure(resident));
        string full=Path.GetFullPath(options.CachePath);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full,JsonSerializer.Serialize(plan,new JsonSerializerOptions {WriteIndented=true}));
        return plan;

        static double[] Measure(StoriesModel model)
        {
            using var session=model.CreateSession();
            float[] logits=new float[32000];double[] samples=new double[5];
            for(int i=0;i<8;i++)
            {
                session.Reset();long start=Stopwatch.GetTimestamp();
                session.TryAppend(1,logits);session.TryAppend(9038,logits);
                double ms=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                if(i>=3) samples[i-3]=ms;
            }
            return samples;
        }
    }
}

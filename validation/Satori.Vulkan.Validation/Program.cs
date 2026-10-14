using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Satori.Backends.Vulkan;

try
{
    if(args.Length==2 && args[0]=="--q8-compare") return Q8ValidationRunner.Run(args[1],false,true);
    if(args.Length==2 && args[0]=="--q8-compare-validation") return Q8ValidationRunner.Run(args[1],true,true);
    if(args.Length==2 && args[0]=="--q8") return Q8ValidationRunner.Run(args[1]);
    string mode="required";
    var options=new VulkanOptions(Validation:true);
    for(int i=0;i<args.Length;i++)
    {
        switch(args[i])
        {
            case "--cpu": mode="cpu"; break;
            case "--auto": mode="auto"; break;
            case "--required": mode="required"; break;
            case "--no-validation": options=options with {Validation=false}; break;
            case "--device" when i+1<args.Length: options=options with {DeviceIndex=int.Parse(args[++i],System.Globalization.CultureInfo.InvariantCulture)}; break;
            default: throw new ArgumentException("Usage: [--cpu | --auto | --required] [--no-validation] [--device index]");
        }
    }
    if(options.Validation && mode!="cpu" && Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC")!="1")
        throw new InvalidOperationException("Set VK_VALIDATION_VALIDATE_SYNC=1 for the validation run.");
    var start=Stopwatch.GetTimestamp();
    VulkanContext? context=null;
    VulkanAvailability? availability=null;
    if(mode!="cpu" && !VulkanContext.TryCreate(options,out context,out availability) && mode=="required")
    {
        Console.Error.WriteLine(JsonSerializer.Serialize(availability));
        return 1;
    }
    double initializationMs=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    if(context?.Info.Software==true && mode=="required") { context.Dispose(); throw new InvalidOperationException("Hardware validation requires a hardware device."); }
    var cases=new List<object>();
    ulong peakScratch=0;
    try
    {
        foreach(int count in new[] {0,1,63,64,65,4097})
        {
            float[] input=new float[count],output=new float[count];
            var times=new List<double>();
            var allocated=new List<long>();
            for(int repeat=0;repeat<9;repeat++)
            {
                for(int i=0;i<count;i++) input[i]=(i%97-48)*0.25f+repeat;
                long before=GC.GetAllocatedBytesForCurrentThread();
                start=Stopwatch.GetTimestamp();
                if(context is null) for(int i=0;i<count;i++) output[i]=2*input[i]+1;
                else context.Transform(input,output);
                double elapsed=Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                long bytes=GC.GetAllocatedBytesForCurrentThread()-before;
                for(int i=0;i<count;i++) if(output[i]!=2*input[i]+1) throw new InvalidDataException($"Transform mismatch at {i}.");
                if(repeat==0) continue;
                times.Add(elapsed); allocated.Add(bytes);
            }
            peakScratch=Math.Max(peakScratch,context?.AllocatedBytes??0);
            cases.Add(new {Count=count,Iterations=times.Count,HostSamplesMs=times,AllocatedBytes=allocated,TransferBytesPerCall=context is null?0:count*8,MatchesCpu=true});
        }
    }
    finally { context?.Dispose(); }
    if(context?.ValidationErrorCount>0) throw new InvalidOperationException("Validation errors occurred, including during disposal.");
    var report=new
    {
        TimestampUtc=DateTimeOffset.UtcNow,Runtime=RuntimeInformation.FrameworkDescription,
        Mode=mode,Options=options,ActualBackend=context is null?"CPU":"Vulkan",Availability=availability,
        Device=context?.Info,LoaderApiVersion=context?.LoaderApiVersion,KhronosLayerApiVersion=context?.ValidationLayerApiVersion,InitializationMs=initializationMs,DeviceScratchAllocationBytes=peakScratch,
        ValidationSync=Environment.GetEnvironmentVariable("VK_VALIDATION_VALIDATE_SYNC"),
        ValidationErrors=context?.ValidationErrorCount,ValidationMessages=context?.ValidationMessages,
        GpuKernelDurationMs=(double?)null,
        TimingScope="Host synchronous upload/dispatch/fence/readback; first call per size excluded; no GPU timestamp measurement. Scratch peak sampled before disposal.",Cases=cases
    };
    string directory="reports/vulkan";
    Directory.CreateDirectory(directory);
    File.WriteAllText(Path.Combine(directory,$"{report.ActualBackend.ToLowerInvariant()}-{mode}.json"),JsonSerializer.Serialize(report,new JsonSerializerOptions{WriteIndented=true,Converters={new JsonStringEnumConverter()}}));
    Console.WriteLine($"{report.ActualBackend}: {cases.Count} workloads verified; validation errors: {report.ValidationErrors?.ToString()??"not enabled"}");
    if(context is null && availability is not null) Console.WriteLine($"CPU fallback: {availability.Reason} ({availability.Operation})");
    return 0;
}
catch(Exception exception) when(exception is ArgumentException or InvalidOperationException or IOException or VulkanException or FormatException or OverflowException)
{
    Console.Error.WriteLine(exception.Message);
    return 1;
}

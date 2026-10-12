using Satori.Formats.Gguf;
using Silk.NET.Vulkan;

namespace Satori.Backends.Vulkan;

/// <summary>Owns a model and its optional Vulkan context. Automatic fallback is initialization-only.</summary>
public sealed class VulkanStoriesRuntime : IDisposable
{
    private readonly VulkanContext? _context;
    private readonly VulkanResidentModel? _resident;
    public VulkanResidentStatistics? ResidentStatistics => _resident?.Statistics;
    /// <summary>Candidate allocations and completed work before returning the selected model.</summary>
    public VulkanDeviceInfo? CandidateDevice { get; }
    public VulkanOffloadStatistics? CandidateStatistics { get; }
    public StoriesExecutionPlan? Plan { get; }
    public bool PlanFromCache { get; }
    public StoriesModel Model { get; }
    public string ActualBackend => _context is null?"CPU":"Vulkan";
    public VulkanAvailability? Availability { get; }
    public VulkanDeviceInfo? Device => _context?.Info;
    public int ValidationErrorCount => _context?.ValidationErrorCount??0;
    public IReadOnlyList<string>? ValidationMessages => _context?.ValidationMessages;
    public VulkanOffloadStatistics? Statistics => _context?.GetOffloadStatistics();

    public VulkanStoriesRuntime(string path, StoriesBackendOptions? options = null)
    {
        options ??= new();
        if(options.Mode==StoriesBackendMode.Auto && options.MatVec is not null) throw new ArgumentException("Custom measurement options require explicit Vulkan selection.",nameof(options));
        if(!Enum.IsDefined(options.Path))throw new ArgumentOutOfRangeException(nameof(options));
        if(options.Path==StoriesVulkanPath.Resident && (options.Mode==StoriesBackendMode.Auto || options.MatVec is not null))throw new ArgumentException("Resident execution requires explicit Vulkan selection without MatVec diagnostics.");
        if (!Enum.IsDefined(options.Mode)) throw new ArgumentOutOfRangeException(nameof(options));
        if (options.Mode==StoriesBackendMode.Cpu) { Model=new(path); return; }
        if (!VulkanContext.TryCreate(options.Vulkan??new(),out var context,out var availability))
        {
            Availability=availability;
            if(options.Mode==StoriesBackendMode.VulkanRequired) throw new InvalidOperationException($"Vulkan required: {availability.Reason} ({availability.Operation}).");
            Model=new(path);
            return;
        }
        CandidateDevice=context!.Info;
        string? identity=null;
        if(options.Mode==StoriesBackendMode.Auto)
        {
            try
            {
                if(options.Vulkan?.Validation==true) throw new ArgumentException("Automatic timing selection requires validation disabled.",nameof(options));
                identity=StoriesExecutionPlanner.GetIdentity(context!);
                Plan=StoriesExecutionPlanner.ReadCache(identity,options.Planner??new());
                PlanFromCache=Plan is not null;
                if(Plan?.SelectedBackend=="CPU")
                {
                    CandidateStatistics=context!.GetOffloadStatistics();
                    context.Dispose();
                    Model=new(path);Availability=availability;return;
                }
            }
            catch { context!.Dispose();throw; }
        }
        try
        {
            if(options.Path==StoriesVulkanPath.Resident){_resident=new(path,context!);Model=new(path,sessionFactory:_resident);}
            else Model=new(path,options.MatVec is null?context:new VulkanMatrixFactory(context!,options.MatVec));
        }
        catch(VulkanException error) when(error.Result==Result.ErrorOutOfDeviceMemory && options.Mode==StoriesBackendMode.Auto)
        {
            context!.Dispose();
            Availability=new(VulkanAvailabilityReason.DeviceMemoryUnavailable,error.Operation,(int)error.Result);
            Model=new(path);
            return;
        }
        catch(NotSupportedException error) when(options.Mode==StoriesBackendMode.Auto)
        {
            context!.Dispose();
            Availability=new(VulkanAvailabilityReason.UnsupportedDevice,error.Message);
            Model=new(path);
            return;
        }
        catch { context!.Dispose(); throw; }
        if(options.Mode==StoriesBackendMode.Auto)
        {
            try
            {
                Plan ??= StoriesExecutionPlanner.Select(path,Model,identity!,options.Planner??new());
                CandidateStatistics=context!.GetOffloadStatistics();
                if(Plan.SelectedBackend=="CPU")
                {
                    Model.Dispose();context!.Dispose();Model=new(path);Availability=availability;return;
                }
            }
            catch { Model.Dispose();context!.Dispose();throw; }
        }
        CandidateStatistics ??= context!.GetOffloadStatistics();
        _context=context;
        Availability=availability;
    }

    public void Dispose()
    {
        Model.Dispose();
        _resident?.Dispose();
        _context?.Dispose();
    }
}

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

    public StoriesVulkanPath? ActualPath=>_context is null?null:_resident is null?StoriesVulkanPath.MatVec:StoriesVulkanPath.Resident;
    public VulkanResidentStatistics? CandidateResidentStatistics {get;}

    public VulkanStoriesRuntime(string path,StoriesBackendOptions? options=null)
    {
        options ??=new();
        if(!Enum.IsDefined(options.Mode)||!Enum.IsDefined(options.Path))throw new ArgumentOutOfRangeException(nameof(options));
        if(options.Mode==StoriesBackendMode.Auto && (options.MatVec is not null || options.Path!=StoriesVulkanPath.MatVec))throw new ArgumentException("Auto selects all compatible paths without custom MatVec options.",nameof(options));
        if(options.Path==StoriesVulkanPath.Resident && options.MatVec is not null)throw new ArgumentException("Resident execution does not use MatVec diagnostics.",nameof(options));
        if(options.Mode==StoriesBackendMode.Cpu){Model=new(path);return;}
        if(!VulkanContext.TryCreate(options.Vulkan??new(),out var created,out var availability))
        {
            Availability=availability;
            if(options.Mode==StoriesBackendMode.VulkanRequired)throw new InvalidOperationException($"Vulkan required: {availability.Reason} ({availability.Operation}).");
            Model=new(path);return;
        }
        var context=created!;CandidateDevice=context.Info;
        StoriesModel? legacy=null,residentModel=null;VulkanResidentModel? weights=null;
        try
        {
            StoriesVulkanPath? selected=options.Path;
            if(options.Mode==StoriesBackendMode.Auto)
            {
                if(options.Vulkan?.Validation==true)throw new ArgumentException("Automatic timing selection requires validation disabled.",nameof(options));
                string identity=StoriesExecutionPlanner.GetIdentity(context);
                Plan=StoriesExecutionPlanner.ReadCache(identity,options.Planner??new());PlanFromCache=Plan is not null;
                if(Plan is not null)selected=Plan.SelectedPath;
                else
                {
                    legacy=new(path,new VulkanMatrixFactory(context,new()));weights=new(path,context);residentModel=new(path,sessionFactory:weights);
                    Plan=StoriesExecutionPlanner.Select(path,legacy,residentModel,identity,options.Planner??new());selected=Plan.SelectedPath;
                }
            }
            if(selected==StoriesVulkanPath.Resident)
            {
                weights ??=new(path,context);residentModel ??=new(path,sessionFactory:weights);
            }
            else if(selected==StoriesVulkanPath.MatVec)
                legacy ??=new(path,options.Mode==StoriesBackendMode.Auto?new VulkanMatrixFactory(context,new()):options.MatVec is null?context:new VulkanMatrixFactory(context,options.MatVec));
            CandidateStatistics=context.GetOffloadStatistics();CandidateResidentStatistics=weights?.Statistics;
            if(selected is null)
            {
                legacy?.Dispose();residentModel?.Dispose();weights?.Dispose();context.Dispose();Model=new(path);Availability=availability;return;
            }
            if(selected==StoriesVulkanPath.Resident)
            {
                legacy?.Dispose();Model=residentModel!;_resident=weights;
            }
            else
            {
                residentModel?.Dispose();weights?.Dispose();Model=legacy!;
            }
            _context=context;Availability=availability;
        }
        catch(VulkanException error) when(error.Result==Result.ErrorOutOfDeviceMemory && options.Mode==StoriesBackendMode.Auto)
        {
            legacy?.Dispose();residentModel?.Dispose();weights?.Dispose();context.Dispose();
            Availability=new(VulkanAvailabilityReason.DeviceMemoryUnavailable,error.Operation,(int)error.Result);Model=new(path);
        }
        catch(NotSupportedException error) when(options.Mode==StoriesBackendMode.Auto)
        {
            legacy?.Dispose();residentModel?.Dispose();weights?.Dispose();context.Dispose();
            Availability=new(VulkanAvailabilityReason.UnsupportedDevice,error.Message);Model=new(path);
        }
        catch {legacy?.Dispose();residentModel?.Dispose();weights?.Dispose();context.Dispose();throw;}
    }
    public void Dispose(){Model.Dispose();_resident?.Dispose();_context?.Dispose();}
}

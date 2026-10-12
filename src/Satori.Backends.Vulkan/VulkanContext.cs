using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Silk.NET.Vulkan;
using Satori.Core;
using Silk.NET.Vulkan.Extensions.EXT;

namespace Satori.Backends.Vulkan;

/// <summary>Single-queue synchronous compute owner. Native handles never escape this object.</summary>
public sealed unsafe class VulkanContext : IDisposable, IQuantizedMatrixFactory
{
    private readonly object _sync = new();
    private readonly HashSet<VulkanQ8Matrix> _q8Matrices = [];
    private VulkanQ8Pipeline? _q8Pipeline, _q8CooperativePipeline;
    private uint _allocationCount;
    private readonly ConcurrentQueue<string> _messages = new();
    private int _validationErrors, _generalErrors;
    private Vk? _api;
    private Instance _instance;
    private PhysicalDevice _physical;
    private Device _device;
    private Silk.NET.Vulkan.Queue _queue;
    private CommandPool _commandPool;
    private CommandBuffer _command;
    private Fence _fence;
    private DescriptorSetLayout _setLayout;
    private DescriptorPool _descriptorPool;
    private DescriptorSet _descriptorSet;
    private PipelineLayout _pipelineLayout;
    private Pipeline _pipeline;
    private ExtDebugUtils? _debug;
    private DebugUtilsMessengerEXT _messenger;
    private PfnDebugUtilsMessengerCallbackEXT _callback;
    private bool _callbackCreated, _disposed, _faulted, _submitted;
    private VulkanBuffer[] _buffers = [];
    private int _capacity;
    private PhysicalDeviceMemoryProperties _memoryProperties;

    private VulkanContext() { }
    internal object Sync => _sync;
    internal CommandBuffer Command => _command;
    internal Vk Api => _api!;
    internal Device Device => _device;
    internal PhysicalDeviceMemoryProperties MemoryProperties => _memoryProperties;
    public VulkanDeviceInfo Info { get; private set; } = null!;
    public uint? ValidationLayerApiVersion { get; private set; }
    public uint LoaderApiVersion { get; private set; }
    public int ValidationErrorCount => Volatile.Read(ref _validationErrors);
    public IReadOnlyList<string> ValidationMessages => _messages.ToArray();
    public ulong AllocatedBytes { get { lock (_sync) return _buffers.Aggregate(0UL,(sum,b)=>sum+b.AllocationBytes)+_q8Matrices.Aggregate(0UL,(sum,m)=>sum+m.AllocatedBytes); } }
    public bool IsFaulted { get { lock (_sync) return _faulted; } }

    public static bool TryCreate(VulkanOptions? options, out VulkanContext? context, out VulkanAvailability availability)
    {
        options ??= new();
        if (options.DeviceIndex < 0) throw new ArgumentOutOfRangeException(nameof(options));
        var candidate = new VulkanContext();
        context = null;
        try
        {
            availability = candidate.Initialize(options);
            if (availability.Reason != VulkanAvailabilityReason.Ready)
            {
                candidate.Dispose(); candidate.ThrowIfValidationFailed(knownUnavailable:true);
                availability=availability with {Diagnostics=candidate.ValidationMessages};
                return false;
            }
            context = candidate;
            return true;
        }
        catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException or FileNotFoundException)
        {
            candidate.Dispose(); candidate.ThrowIfValidationFailed(knownUnavailable:true);
            availability = new(VulkanAvailabilityReason.LoaderUnavailable,exception.GetType().Name,Diagnostics:candidate.ValidationMessages);
            return false;
        }
        catch (VulkanException exception) when (exception.Result is Result.ErrorInitializationFailed or Result.ErrorIncompatibleDriver
            or Result.ErrorFeatureNotPresent or Result.ErrorExtensionNotPresent or Result.ErrorLayerNotPresent or Result.ErrorOutOfDeviceMemory)
        {
            candidate.Dispose(); candidate.ThrowIfValidationFailed(knownUnavailable:true);
            availability = new(exception.Result==Result.ErrorOutOfDeviceMemory ? VulkanAvailabilityReason.DeviceMemoryUnavailable : VulkanAvailabilityReason.InitializationFailed,
                exception.Operation,(int)exception.Result,candidate.ValidationMessages);
            return false;
        }
        catch { candidate.Dispose(); candidate.ThrowIfValidationFailed(); throw; }
    }

    private VulkanAvailability Initialize(VulkanOptions options)
    {
        _api = Vk.GetApi();
        uint version=0;
        Check(Api.EnumerateInstanceVersion(&version),"EnumerateInstanceVersion");
        LoaderApiVersion=version;
        if (version < Vk.MakeVersion(1,1,0)) return new(VulkanAvailabilityReason.UnsupportedDevice,"Loader API < 1.1");
        if (options.Validation && (!HasLayer("VK_LAYER_KHRONOS_validation") || !HasExtension("VK_EXT_debug_utils")))
            return new(VulkanAvailabilityReason.ValidationUnavailable,"Khronos validation layer/debug utils absent");
        fixed (byte* applicationName = "Satori\0"u8)
        fixed (byte* layerName = "VK_LAYER_KHRONOS_validation\0"u8)
        fixed (byte* extensionName = "VK_EXT_debug_utils\0"u8)
        {
            byte* layer = layerName; byte* extension = extensionName;
            var application = new ApplicationInfo {SType=StructureType.ApplicationInfo,PApplicationName=applicationName,ApiVersion=Vk.MakeVersion(1,1,0)};
            var debugInfo = new DebugUtilsMessengerCreateInfoEXT();
            if (options.Validation)
            {
                _callback = new((DebugUtilsMessengerCallbackFunctionEXT)DebugCallback); _callbackCreated = true;
                debugInfo = new() {SType=StructureType.DebugUtilsMessengerCreateInfoExt,
                    MessageSeverity=DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt|DebugUtilsMessageSeverityFlagsEXT.WarningBitExt,
                    MessageType=DebugUtilsMessageTypeFlagsEXT.GeneralBitExt|DebugUtilsMessageTypeFlagsEXT.ValidationBitExt|DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt,
                    PfnUserCallback=_callback};
            }
            var create = new InstanceCreateInfo {SType=StructureType.InstanceCreateInfo,PApplicationInfo=&application,
                EnabledLayerCount=options.Validation?1u:0, PpEnabledLayerNames=options.Validation?&layer:null,
                EnabledExtensionCount=options.Validation?1u:0,PpEnabledExtensionNames=options.Validation?&extension:null,
                PNext=options.Validation?&debugInfo:null};
            Check(Api.CreateInstance(in create,null,out _instance),"CreateInstance");
            if (options.Validation)
            {
                if (!Api.TryGetInstanceExtension(_instance,out _debug)) return new(VulkanAvailabilityReason.ValidationUnavailable,"Load debug utils");
                Check(_debug!.CreateDebugUtilsMessenger(_instance,in debugInfo,null,out _messenger),"CreateDebugUtilsMessenger");
            }
        }
        uint count = 0;
        Check(Api.EnumeratePhysicalDevices(_instance,&count,null),"EnumeratePhysicalDevices count");
        if (count==0) return new(VulkanAvailabilityReason.NoDevice,"EnumeratePhysicalDevices");
        var devices = new PhysicalDevice[count];
        fixed (PhysicalDevice* pointer=devices) Check(Api.EnumeratePhysicalDevices(_instance,&count,pointer),"EnumeratePhysicalDevices");
        if (options.DeviceIndex >= count) return new(VulkanAvailabilityReason.UnsupportedDevice,"Device index not enumerated");
        for (int i=0;i<count;i++)
        {
            if (options.DeviceIndex is int selected && selected!=i) continue;
            if (SelectDevice(devices[i],i)) break;
        }
        if (_physical.Handle==0) return new(VulkanAvailabilityReason.UnsupportedDevice,"No eligible compute device");
        float priority = 1;
        var queueCreate = new DeviceQueueCreateInfo {SType=StructureType.DeviceQueueCreateInfo,QueueFamilyIndex=Info.QueueFamily,QueueCount=1,PQueuePriorities=&priority};
        var deviceCreate = new DeviceCreateInfo {SType=StructureType.DeviceCreateInfo,QueueCreateInfoCount=1,PQueueCreateInfos=&queueCreate};
        Check(Api.CreateDevice(_physical,in deviceCreate,null,out _device),"CreateDevice");
        Api.GetDeviceQueue(_device,Info.QueueFamily,0,out _queue);
        var poolCreate = new CommandPoolCreateInfo {SType=StructureType.CommandPoolCreateInfo,QueueFamilyIndex=Info.QueueFamily,Flags=CommandPoolCreateFlags.ResetCommandBufferBit};
        Check(Api.CreateCommandPool(_device,in poolCreate,null,out _commandPool),"CreateCommandPool");
        var commandAllocate = new CommandBufferAllocateInfo {SType=StructureType.CommandBufferAllocateInfo,CommandPool=_commandPool,Level=CommandBufferLevel.Primary,CommandBufferCount=1};
        Check(Api.AllocateCommandBuffers(_device,in commandAllocate,out _command),"AllocateCommandBuffers");
        var fenceCreate = new FenceCreateInfo {SType=StructureType.FenceCreateInfo};
        Check(Api.CreateFence(_device,in fenceCreate,null,out _fence),"CreateFence");
        CreatePipeline();
        ThrowIfValidationFailed();
        return new(VulkanAvailabilityReason.Ready,"Ready");
    }

    private bool HasLayer(string name)
    {
        uint count=0;
        Check(Api.EnumerateInstanceLayerProperties(&count,null),"EnumerateInstanceLayerProperties count");
        var layers=new LayerProperties[count];
        fixed (LayerProperties* pointer=layers)
        {
            Check(Api.EnumerateInstanceLayerProperties(&count,pointer),"EnumerateInstanceLayerProperties");
            for(int i=0;i<count;i++) if(Marshal.PtrToStringUTF8((nint)pointer[i].LayerName)==name) { ValidationLayerApiVersion=pointer[i].SpecVersion; return true; }
        }
        return false;
    }

    private bool HasExtension(string name)
    {
        uint count=0;
        Check(Api.EnumerateInstanceExtensionProperties((byte*)null,&count,null),"EnumerateInstanceExtensionProperties count");
        var extensions=new ExtensionProperties[count];
        fixed (ExtensionProperties* pointer=extensions)
        {
            Check(Api.EnumerateInstanceExtensionProperties((byte*)null,&count,pointer),"EnumerateInstanceExtensionProperties");
            for(int i=0;i<count;i++) if(Marshal.PtrToStringUTF8((nint)pointer[i].ExtensionName)==name) return true;
        }
        return false;
    }

    private bool SelectDevice(PhysicalDevice physical, int index)
    {
        Api.GetPhysicalDeviceProperties(physical,out var properties);
        var limits=properties.Limits;
        if(properties.ApiVersion<Vk.MakeVersion(1,1,0) || limits.MaxComputeWorkGroupInvocations<64 || limits.MaxComputeWorkGroupSize[0]<64
            || limits.MaxPushConstantsSize<4 || limits.MaxPerStageDescriptorStorageBuffers<2 || limits.MaxDescriptorSetStorageBuffers<2
            || limits.MaxStorageBufferRange<sizeof(float) || limits.MaxMemoryAllocationCount<4) return false;
        uint count=0;
        Api.GetPhysicalDeviceQueueFamilyProperties(physical,&count,null);
        var families=new QueueFamilyProperties[count];
        fixed(QueueFamilyProperties* pointer=families) Api.GetPhysicalDeviceQueueFamilyProperties(physical,&count,pointer);
        for(uint family=0;family<count;family++)
        {
            if(families[family].QueueCount==0 || (families[family].QueueFlags&QueueFlags.ComputeBit)==0) continue;
            var maintenance = new PhysicalDeviceMaintenance3Properties {SType=StructureType.PhysicalDeviceMaintenance3Properties};
            var id = new PhysicalDeviceIDProperties {SType=StructureType.PhysicalDeviceIDProperties,PNext=&maintenance};
            var extended = new PhysicalDeviceProperties2 {SType=StructureType.PhysicalDeviceProperties2,PNext=&id};
            Api.GetPhysicalDeviceProperties2(physical,&extended);
            Api.GetPhysicalDeviceMemoryProperties(physical,out _memoryProperties);
            uint[] flags=Enumerable.Range(0,(int)_memoryProperties.MemoryTypeCount).Select(i=>(uint)_memoryProperties.MemoryTypes[i].PropertyFlags).ToArray();
            if(!flags.Any(f=>(f&(uint)MemoryPropertyFlags.DeviceLocalBit)!=0) || !flags.Any(f=>(f&(uint)MemoryPropertyFlags.HostVisibleBit)!=0)) continue;
            Info=new(index,Marshal.PtrToStringUTF8((nint)properties.DeviceName)!,Convert.ToHexString(new ReadOnlySpan<byte>(id.DeviceUuid,16)).ToLowerInvariant(),
                properties.VendorID,properties.DeviceID,properties.ApiVersion,properties.DriverVersion,properties.DeviceType==PhysicalDeviceType.Cpu,
                family,families[family].TimestampValidBits,limits.TimestampPeriod,limits.MaxStorageBufferRange,limits.MaxComputeWorkGroupCount[0],
                limits.MaxComputeWorkGroupInvocations,limits.MaxPushConstantsSize,Math.Min(limits.MaxPerStageDescriptorStorageBuffers,limits.MaxDescriptorSetStorageBuffers),limits.MaxMemoryAllocationCount,maintenance.MaxMemoryAllocationSize,limits.MinStorageBufferOffsetAlignment,limits.NonCoherentAtomSize,Array.AsReadOnly(flags));
            _physical=physical;
            return true;
        }
        return false;
    }

    private uint DebugCallback(DebugUtilsMessageSeverityFlagsEXT severity, DebugUtilsMessageTypeFlagsEXT type, DebugUtilsMessengerCallbackDataEXT* data, void* user)
    {
        if((severity&DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt)!=0)
        {
            if((type&(DebugUtilsMessageTypeFlagsEXT.ValidationBitExt|DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt))!=0)
                Interlocked.Increment(ref _validationErrors);
            else Interlocked.Increment(ref _generalErrors);
        }
        try { _messages.Enqueue($"{severity}/{type}: {Marshal.PtrToStringUTF8((nint)data->PMessage)}"); }
        catch { /* Exceptions cannot cross the native callback. */ }
        return 0;
    }

    internal void Check(Result result, string operation)
    {
        if(result==Result.Success) return;
        if(result==Result.ErrorDeviceLost) _faulted=true;
        throw new VulkanException(operation,result);
    }

    internal void ThrowIfValidationFailed(bool knownUnavailable = false)
    {
        if(ValidationErrorCount==0 && (knownUnavailable || Volatile.Read(ref _generalErrors)==0)) return;
        _faulted=true;
        throw new InvalidOperationException("Vulkan validation reported an error. See ValidationMessages.");
    }

    internal uint FindMemoryType(uint bits, MemoryPropertyFlags required)
    {
        // Prefer coherent staging, while supporting non-coherent host-visible heaps.
        for(int pass=0;pass<2;pass++)
        for(uint i=0;i<_memoryProperties.MemoryTypeCount;i++)
        {
            var flags=_memoryProperties.MemoryTypes[(int)i].PropertyFlags;
            if((bits&(1u<<(int)i))==0 || (flags&required)!=required) continue;
            if(pass==0 && (required&MemoryPropertyFlags.HostVisibleBit)!=0 && (flags&MemoryPropertyFlags.HostCoherentBit)==0) continue;
            return i;
        }
        throw new VulkanException("Find compatible memory type",Result.ErrorFeatureNotPresent);
    }

    private void CreatePipeline()
    {
        var bindings=stackalloc DescriptorSetLayoutBinding[2];
        for(uint i=0;i<2;i++) bindings[i]=new() {Binding=i,DescriptorType=DescriptorType.StorageBuffer,DescriptorCount=1,StageFlags=ShaderStageFlags.ComputeBit};
        var setCreate=new DescriptorSetLayoutCreateInfo {SType=StructureType.DescriptorSetLayoutCreateInfo,BindingCount=2,PBindings=bindings};
        Check(Api.CreateDescriptorSetLayout(_device,in setCreate,null,out _setLayout),"CreateDescriptorSetLayout");
        var setLayout=_setLayout;
        var push=new PushConstantRange {StageFlags=ShaderStageFlags.ComputeBit,Offset=0,Size=4};
        var layoutCreate=new PipelineLayoutCreateInfo {SType=StructureType.PipelineLayoutCreateInfo,SetLayoutCount=1,PSetLayouts=&setLayout,PushConstantRangeCount=1,PPushConstantRanges=&push};
        Check(Api.CreatePipelineLayout(_device,in layoutCreate,null,out _pipelineLayout),"CreatePipelineLayout");
        var size=new DescriptorPoolSize {Type=DescriptorType.StorageBuffer,DescriptorCount=2};
        var poolCreate=new DescriptorPoolCreateInfo {SType=StructureType.DescriptorPoolCreateInfo,MaxSets=1,PoolSizeCount=1,PPoolSizes=&size};
        Check(Api.CreateDescriptorPool(_device,in poolCreate,null,out _descriptorPool),"CreateDescriptorPool");
        var allocate=new DescriptorSetAllocateInfo {SType=StructureType.DescriptorSetAllocateInfo,DescriptorPool=_descriptorPool,DescriptorSetCount=1,PSetLayouts=&setLayout};
        Check(Api.AllocateDescriptorSets(_device,in allocate,out _descriptorSet),"AllocateDescriptorSets");
        byte[] bytes=VulkanShader.Load();
        ShaderModule shader=default;
        try
        {
            fixed(byte* code=bytes)
            {
                var shaderCreate=new ShaderModuleCreateInfo {SType=StructureType.ShaderModuleCreateInfo,CodeSize=(nuint)bytes.Length,PCode=(uint*)code};
                Check(Api.CreateShaderModule(_device,in shaderCreate,null,out shader),"CreateShaderModule");
            }
            fixed(byte* entry="main\0"u8)
            {
                var create=new ComputePipelineCreateInfo {SType=StructureType.ComputePipelineCreateInfo,Layout=_pipelineLayout,
                    Stage=new() {SType=StructureType.PipelineShaderStageCreateInfo,Stage=ShaderStageFlags.ComputeBit,Module=shader,PName=entry}};
                Check(Api.CreateComputePipelines(_device,default,1,in create,null,out _pipeline),"CreateComputePipelines");
            }
        }
        finally { if(shader.Handle!=0) Api.DestroyShaderModule(_device,shader,null); }
    }

    /// <summary>Writes y = 2*x + 1 after successful GPU completion. Input/output must not overlap.</summary>
    public void Transform(ReadOnlySpan<float> input, Span<float> output)
    {
        lock(_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            if(_faulted) throw new InvalidOperationException("Vulkan context is faulted.");
            if(output.Length!=input.Length || input.Overlaps(output)) throw new ArgumentException("Expected equally sized, non-overlapping vectors.");
            if(input.IsEmpty) return;
            ulong bytes=(ulong)input.Length*sizeof(float);
            uint groups=((uint)input.Length+63)/64;
            if(bytes>Info.MaxStorageBufferRange || groups>Info.MaxDispatchGroups) throw new ArgumentOutOfRangeException(nameof(input),"Workload exceeds Vulkan limits.");
            EnsureCapacity(input.Length);
            _buffers[0].Write(input);
            BeginCommands();
            var copy=new BufferCopy {Size=bytes};
            Api.CmdCopyBuffer(_command,_buffers[0].Handle,_buffers[1].Handle,1,in copy);
            Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderReadBit);
            Api.CmdBindPipeline(_command,PipelineBindPoint.Compute,_pipeline);
            var set=_descriptorSet;
            Api.CmdBindDescriptorSets(_command,PipelineBindPoint.Compute,_pipelineLayout,0,1,in set,0,null);
            uint count=(uint)input.Length;
            Api.CmdPushConstants(_command,_pipelineLayout,ShaderStageFlags.ComputeBit,0,4,&count);
            Api.CmdDispatch(_command,groups,1,1);
            Barrier(PipelineStageFlags.ComputeShaderBit,AccessFlags.ShaderWriteBit,PipelineStageFlags.TransferBit,AccessFlags.TransferReadBit);
            Api.CmdCopyBuffer(_command,_buffers[2].Handle,_buffers[3].Handle,1,in copy);
            Barrier(PipelineStageFlags.TransferBit,AccessFlags.TransferWriteBit,PipelineStageFlags.HostBit,AccessFlags.HostReadBit);
            SubmitAndWait();
            _buffers[3].Read(output);
        }
    }

    internal void Barrier(PipelineStageFlags sourceStage, AccessFlags sourceAccess, PipelineStageFlags destinationStage, AccessFlags destinationAccess)
    {
        var barrier=new MemoryBarrier {SType=StructureType.MemoryBarrier,SrcAccessMask=sourceAccess,DstAccessMask=destinationAccess};
        Api.CmdPipelineBarrier(_command,sourceStage,destinationStage,0,1,in barrier,0,null,0,null);
    }

    internal void EnsureReady()
    {
        ObjectDisposedException.ThrowIf(_disposed,this);
        if(_faulted) throw new InvalidOperationException("Vulkan context is faulted.");
    }

    internal void ReserveAllocation()
    {
        if(_allocationCount>=Info.MaxMemoryAllocations) throw new VulkanException("Vulkan allocation count limit",Result.ErrorOutOfDeviceMemory);
        _allocationCount++;
    }

    internal void ReleaseAllocation() => _allocationCount--;

    internal void BeginCommands()
    {
        Check(Api.ResetCommandBuffer(_command,0),"ResetCommandBuffer");
        var begin=new CommandBufferBeginInfo {SType=StructureType.CommandBufferBeginInfo,Flags=CommandBufferUsageFlags.OneTimeSubmitBit};
        Check(Api.BeginCommandBuffer(_command,in begin),"BeginCommandBuffer");
    }

    internal VulkanSubmissionTiming SubmitAndWait(bool measure = false)
    {
        long start=measure?Stopwatch.GetTimestamp():0;
        Check(Api.EndCommandBuffer(_command),"EndCommandBuffer");
        ThrowIfValidationFailed();
        var fence=_fence;
        Check(Api.ResetFences(_device,1,in fence),"ResetFences");
        var command=_command;
        var submit=new SubmitInfo {SType=StructureType.SubmitInfo,CommandBufferCount=1,PCommandBuffers=&command};
        long ready=measure?Stopwatch.GetTimestamp():0;
        long submitted=0,finished=0;
        try
        {
            Check(Api.QueueSubmit(_queue,1,in submit,_fence),"QueueSubmit");
            _submitted=true;
            submitted=measure?Stopwatch.GetTimestamp():0;
            Check(Api.WaitForFences(_device,1,in fence,true,ulong.MaxValue),"WaitForFences");
            _submitted=false;
            finished=measure?Stopwatch.GetTimestamp():0;
        }
        catch { _faulted=true; throw; }
        ThrowIfValidationFailed();
        if(!measure) return default;
        long end=Stopwatch.GetTimestamp();
        return new(Stopwatch.GetElapsedTime(start,ready).TotalMilliseconds,Stopwatch.GetElapsedTime(ready,submitted).TotalMilliseconds,
            Stopwatch.GetElapsedTime(submitted,finished).TotalMilliseconds,Stopwatch.GetElapsedTime(finished,end).TotalMilliseconds);
    }

    IQuantizedMatrix IQuantizedMatrixFactory.Create(ReadOnlySpan<byte> payload, QuantizedMatrixShape shape) =>
        CreateQ8Matrix(payload,shape,new(MeasureKernelTime:true));

    public VulkanOffloadStatistics GetOffloadStatistics()
    {
        lock(_sync)
        {
            return new(_q8Matrices.Count,_q8Matrices.Aggregate(0UL,(sum,m)=>sum+m.AllocatedBytes),
                _q8Matrices.Aggregate(0UL,(sum,m)=>sum+m.WeightAllocationBytes),
                _q8Matrices.Sum(m=>m.CompletedDispatches),_q8Matrices.Sum(m=>m.TransferBytes),
                _q8Matrices.Sum(m=>m.HostMilliseconds),_q8Matrices.Sum(m=>m.CompletedDispatches)>0 && _q8Matrices.Sum(m=>m.KernelSamples)==_q8Matrices.Sum(m=>m.CompletedDispatches)?_q8Matrices.Sum(m=>m.KernelMilliseconds):null,
                _q8Matrices.Aggregate(default(VulkanHostPhases),(sum,m)=>sum+m.HostPhases),_q8Matrices.Sum(m=>m.PhaseSamples));
        }
    }

    public VulkanQ8Matrix CreateQ8Matrix(ReadOnlySpan<byte> payload, QuantizedMatrixShape shape, VulkanQ8Options? options = null)
    {
        lock(_sync)
        {
            EnsureReady();
            if(shape.Format!=BlockQuantization.Q8_0) throw new NotSupportedException("Only Q8_0 is supported by this kernel.");
            QuantizedMatVec.ValidatePayload(payload,shape);
            ulong paddedBytes=((ulong)payload.Length+3)&~3UL;
            if(payload.Length>int.MaxValue-3 || paddedBytes>Info.MaxStorageBufferRange || (ulong)shape.Columns*4>Info.MaxStorageBufferRange
                || (ulong)shape.Rows*4>Info.MaxStorageBufferRange || ((ulong)shape.Rows+63)/64>Info.MaxDispatchGroups)
                throw new ArgumentOutOfRangeException(nameof(shape),"Matrix exceeds Vulkan buffer or dispatch limits.");
            if(Info.MaxStorageDescriptors<3 || Info.MaxPushConstantsSize<8) throw new NotSupportedException("Q8_0 requires three storage descriptors and eight push-constant bytes.");
            options ??= new();
            if(!Enum.IsDefined(options.Kernel)) throw new ArgumentOutOfRangeException(nameof(options));
            if(options.Kernel==VulkanQ8Kernel.Cooperative64 && (ulong)shape.Rows>Info.MaxDispatchGroups) throw new ArgumentOutOfRangeException(nameof(shape));
            var pipeline=options.Kernel==VulkanQ8Kernel.Cooperative64?(_q8CooperativePipeline ??= new(this,options.Kernel)):(_q8Pipeline ??= new(this));
            var matrix=new VulkanQ8Matrix(this,pipeline,payload,shape,options);
            try { _q8Matrices.Add(matrix); return matrix; }
            catch { matrix.DisposeCore(); throw; }
        }
    }

    internal void Forget(VulkanQ8Matrix matrix) => _q8Matrices.Remove(matrix);

    private void EnsureCapacity(int count)
    {
        if(count<=_capacity) return;
        // Free completed scratch before replacement to respect the four-allocation baseline.
        foreach(var buffer in _buffers) buffer.Dispose();
        _buffers=[]; _capacity=0;
        var created=new List<VulkanBuffer>(4);
        ulong bytes=(ulong)count*sizeof(float);
        try
        {
            created.Add(new(this,bytes,BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.HostVisibleBit));
            created.Add(new(this,bytes,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.DeviceLocalBit));
            created.Add(new(this,bytes,BufferUsageFlags.StorageBufferBit|BufferUsageFlags.TransferSrcBit,MemoryPropertyFlags.DeviceLocalBit));
            created.Add(new(this,bytes,BufferUsageFlags.TransferDstBit,MemoryPropertyFlags.HostVisibleBit));
            _buffers=created.ToArray(); _capacity=count;
            var infos=stackalloc DescriptorBufferInfo[2];
            var writes=stackalloc WriteDescriptorSet[2];
            for(uint i=0;i<2;i++)
            {
                infos[i]=new() {Buffer=_buffers[i+1].Handle,Offset=0,Range=bytes};
                writes[i]=new() {SType=StructureType.WriteDescriptorSet,DstSet=_descriptorSet,DstBinding=i,DescriptorCount=1,DescriptorType=DescriptorType.StorageBuffer,PBufferInfo=&infos[i]};
            }
            Api.UpdateDescriptorSets(_device,2,writes,0,null);
            ThrowIfValidationFailed();
        }
        catch
        {
            foreach(var buffer in created) buffer.Dispose();
            _buffers=[]; _capacity=0;
            throw;
        }
    }

    internal void CompletePendingForDisposal()
    {
        if(!_submitted) return;
        var result=Api.DeviceWaitIdle(_device);
        if(result!=Result.Success && result!=Result.ErrorDeviceLost) throw new VulkanException("DeviceWaitIdle teardown",result);
        _submitted=false;
    }

    public void Dispose()
    {
        lock(_sync)
        {
            if(_disposed) return;
            _disposed=true;
            if(_api is null) return;
            if(_device.Handle!=0)
            {
                CompletePendingForDisposal();
                foreach(var matrix in _q8Matrices) matrix.DisposeCore();
                _q8Matrices.Clear();
                _q8Pipeline?.Dispose();
                _q8CooperativePipeline?.Dispose();
                if(_commandPool.Handle!=0) Api.DestroyCommandPool(_device,_commandPool,null);
                if(_descriptorPool.Handle!=0) Api.DestroyDescriptorPool(_device,_descriptorPool,null);
                if(_pipeline.Handle!=0) Api.DestroyPipeline(_device,_pipeline,null);
                if(_pipelineLayout.Handle!=0) Api.DestroyPipelineLayout(_device,_pipelineLayout,null);
                if(_setLayout.Handle!=0) Api.DestroyDescriptorSetLayout(_device,_setLayout,null);
                foreach(var buffer in _buffers) buffer.Dispose();
                _buffers=[];
                if(_fence.Handle!=0) Api.DestroyFence(_device,_fence,null);
                Api.DestroyDevice(_device,null);
            }
            if(_messenger.Handle!=0) _debug!.DestroyDebugUtilsMessenger(_instance,_messenger,null);
            if(_instance.Handle!=0) Api.DestroyInstance(_instance,null);
            _debug?.Dispose();
            if(_callbackCreated) _callback.Dispose();
            _api.Dispose();
        }
    }
}

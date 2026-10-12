using Silk.NET.Vulkan;
using System.Runtime.InteropServices;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace Satori.Backends.Vulkan;

/// <summary>Dedicated allocation; all access and destruction are serialized by its context.</summary>
internal sealed unsafe class VulkanBuffer : IDisposable
{
    private readonly VulkanContext _context;
    internal VkBuffer Handle;
    internal DeviceMemory Memory;
    internal ulong AllocationBytes;
    internal bool Coherent;
    private bool _reserved;

    internal VulkanBuffer(VulkanContext context, ulong bytes, BufferUsageFlags usage, MemoryPropertyFlags required)
    {
        _context = context;
        try
        {
            var info = new BufferCreateInfo { SType=StructureType.BufferCreateInfo, Size=bytes, Usage=usage, SharingMode=SharingMode.Exclusive };
            context.Check(context.Api.CreateBuffer(context.Device,in info,null,out Handle),"CreateBuffer");
            context.Api.GetBufferMemoryRequirements(context.Device,Handle,out var requirements);
            uint type = context.FindMemoryType(requirements.MemoryTypeBits,required);
            if (requirements.Size>context.Info.MaxAllocationBytes) throw new ArgumentOutOfRangeException(nameof(bytes),"Allocation exceeds Vulkan maximum.");
            MemoryPropertyFlags flags = context.MemoryProperties.MemoryTypes[(int)type].PropertyFlags;
            Coherent = (flags&MemoryPropertyFlags.HostCoherentBit)!=0;
            AllocationBytes = requirements.Size;
            var allocation = new MemoryAllocateInfo {SType=StructureType.MemoryAllocateInfo,AllocationSize=requirements.Size,MemoryTypeIndex=type};
            context.ReserveAllocation(); _reserved=true;
            context.Check(context.Api.AllocateMemory(context.Device,in allocation,null,out Memory),"AllocateMemory");
            context.Check(context.Api.BindBufferMemory(context.Device,Handle,Memory,0),"BindBufferMemory");
        }
        catch { Dispose(); throw; }
    }

    internal void Write(ReadOnlySpan<float> input) => WriteBytes(MemoryMarshal.AsBytes(input));

    internal void WriteBytes(ReadOnlySpan<byte> input, int padding = 0)
    {
        void* pointer;
        _context.Check(_context.Api.MapMemory(_context.Device,Memory,0,AllocationBytes,0,&pointer),"MapMemory upload");
        try
        {
            input.CopyTo(new Span<byte>(pointer,input.Length));
            new Span<byte>((byte*)pointer+input.Length,padding).Clear();
            if (!Coherent)
            {
                var range = new MappedMemoryRange {SType=StructureType.MappedMemoryRange,Memory=Memory,Offset=0,Size=Vk.WholeSize};
                _context.Check(_context.Api.FlushMappedMemoryRanges(_context.Device,1,in range),"FlushMappedMemoryRanges");
            }
        }
        finally { _context.Api.UnmapMemory(_context.Device,Memory); }
    }

    internal void Read(Span<float> output)
    {
        void* pointer;
        _context.Check(_context.Api.MapMemory(_context.Device,Memory,0,AllocationBytes,0,&pointer),"MapMemory readback");
        try
        {
            if (!Coherent)
            {
                var range = new MappedMemoryRange {SType=StructureType.MappedMemoryRange,Memory=Memory,Offset=0,Size=Vk.WholeSize};
                _context.Check(_context.Api.InvalidateMappedMemoryRanges(_context.Device,1,in range),"InvalidateMappedMemoryRanges");
            }
            _context.ThrowIfValidationFailed();
            new ReadOnlySpan<float>(pointer,output.Length).CopyTo(output);
        }
        finally { _context.Api.UnmapMemory(_context.Device,Memory); }
    }

    public void Dispose()
    {
        _context.CompletePendingForDisposal();
        if (Handle.Handle!=0) { _context.Api.DestroyBuffer(_context.Device,Handle,null); Handle=default; }
        if (Memory.Handle!=0) { _context.Api.FreeMemory(_context.Device,Memory,null); Memory=default; }
        if (_reserved) { _context.ReleaseAllocation(); _reserved=false; }
    }
}

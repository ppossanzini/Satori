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
    private void* _mapped;
    private readonly ulong _bytes;

    internal VulkanBuffer(VulkanContext context, ulong bytes, BufferUsageFlags usage, MemoryPropertyFlags required)
    {
        _context = context; _bytes=bytes;
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
            if((required&MemoryPropertyFlags.HostVisibleBit)!=0)
            {
                void* pointer;
                context.Check(context.Api.MapMemory(context.Device,Memory,0,AllocationBytes,0,&pointer),"Map persistent staging");
                _mapped=pointer;
            }
        }
        catch { Dispose(); throw; }
    }

    internal void Write(ReadOnlySpan<float> input) => WriteBytes(MemoryMarshal.AsBytes(input));

    internal void WriteBytes(ReadOnlySpan<byte> input, int padding = 0)
    {
        if(_mapped is null) throw new InvalidOperationException("Buffer is not host mapped.");
        if(padding<0 || (ulong)input.Length+(ulong)padding>_bytes) throw new ArgumentOutOfRangeException(nameof(input));
        input.CopyTo(new Span<byte>(_mapped,input.Length));
        new Span<byte>((byte*)_mapped+input.Length,padding).Clear();
        if (!Coherent)
        {
            var range = new MappedMemoryRange {SType=StructureType.MappedMemoryRange,Memory=Memory,Offset=0,Size=Vk.WholeSize};
            _context.Check(_context.Api.FlushMappedMemoryRanges(_context.Device,1,in range),"FlushMappedMemoryRanges");
        }
    }

    internal void Read(Span<float> output)
    {
        if(_mapped is null) throw new InvalidOperationException("Buffer is not host mapped.");
        if((ulong)output.Length*4>_bytes) throw new ArgumentOutOfRangeException(nameof(output));
        if (!Coherent)
        {
            var range = new MappedMemoryRange {SType=StructureType.MappedMemoryRange,Memory=Memory,Offset=0,Size=Vk.WholeSize};
            _context.Check(_context.Api.InvalidateMappedMemoryRanges(_context.Device,1,in range),"InvalidateMappedMemoryRanges");
        }
        _context.ThrowIfValidationFailed();
        new ReadOnlySpan<float>(_mapped,output.Length).CopyTo(output);
    }

    public void Dispose()
    {
        _context.CompletePendingForDisposal();
        if (Handle.Handle!=0) { _context.Api.DestroyBuffer(_context.Device,Handle,null); Handle=default; }
        if (_mapped is not null) { _context.Api.UnmapMemory(_context.Device,Memory); _mapped=null; }
        if (Memory.Handle!=0) { _context.Api.FreeMemory(_context.Device,Memory,null); Memory=default; }
        if (_reserved) { _context.ReleaseAllocation(); _reserved=false; }
    }
}

using Silk.NET.Vulkan;

namespace Satori.Backends.Vulkan;

/// <summary>Shared compressed-weight compute pipeline; descriptor sets belong to matrices.</summary>
internal sealed unsafe class VulkanQ8Pipeline : IDisposable
{
    private readonly VulkanContext _context;
    internal DescriptorSetLayout SetLayout;
    internal PipelineLayout Layout;
    internal Pipeline Handle;
    internal VulkanQ8Kernel Kernel { get; }

    internal VulkanQ8Pipeline(VulkanContext context, VulkanQ8Kernel kernel = VulkanQ8Kernel.Row64, bool resident=false)
    {
        _context=context; Kernel=kernel;
        ShaderModule shader=default;
        try
        {
            var bindings=stackalloc DescriptorSetLayoutBinding[3];
            for(uint i=0;i<3;i++) bindings[i]=new() {Binding=i,DescriptorType=DescriptorType.StorageBuffer,DescriptorCount=1,StageFlags=ShaderStageFlags.ComputeBit};
            var createSet=new DescriptorSetLayoutCreateInfo {SType=StructureType.DescriptorSetLayoutCreateInfo,BindingCount=3,PBindings=bindings};
            context.Check(context.Api.CreateDescriptorSetLayout(context.Device,in createSet,null,out SetLayout),"Create Q8 set layout");
            var set=SetLayout;
            var push=new PushConstantRange {StageFlags=ShaderStageFlags.ComputeBit,Offset=0,Size=resident?16u:8u};
            var createLayout=new PipelineLayoutCreateInfo {SType=StructureType.PipelineLayoutCreateInfo,SetLayoutCount=1,PSetLayouts=&set,PushConstantRangeCount=1,PPushConstantRanges=&push};
            context.Check(context.Api.CreatePipelineLayout(context.Device,in createLayout,null,out Layout),"Create Q8 pipeline layout");
            byte[] bytes=VulkanShader.Load((kernel==VulkanQ8Kernel.Cooperative64?"q8_cooperative":"q8_matvec")+(resident?"_resident":""));
            fixed(byte* code=bytes)
            {
                var createShader=new ShaderModuleCreateInfo {SType=StructureType.ShaderModuleCreateInfo,CodeSize=(nuint)bytes.Length,PCode=(uint*)code};
                context.Check(context.Api.CreateShaderModule(context.Device,in createShader,null,out shader),"Create Q8 shader");
            }
            fixed(byte* entry="main\0"u8)
            {
                var create=new ComputePipelineCreateInfo {SType=StructureType.ComputePipelineCreateInfo,Layout=Layout,
                    Stage=new() {SType=StructureType.PipelineShaderStageCreateInfo,Stage=ShaderStageFlags.ComputeBit,Module=shader,PName=entry}};
                context.Check(context.Api.CreateComputePipelines(context.Device,default,1,in create,null,out Handle),"Create Q8 pipeline");
            }
            context.ThrowIfValidationFailed();
        }
        catch { Dispose(); throw; }
        finally { if(shader.Handle!=0) context.Api.DestroyShaderModule(context.Device,shader,null); }
    }

    public void Dispose()
    {
        if(Handle.Handle!=0) { _context.Api.DestroyPipeline(_context.Device,Handle,null); Handle=default; }
        if(Layout.Handle!=0) { _context.Api.DestroyPipelineLayout(_context.Device,Layout,null); Layout=default; }
        if(SetLayout.Handle!=0) { _context.Api.DestroyDescriptorSetLayout(_context.Device,SetLayout,null); SetLayout=default; }
    }
}

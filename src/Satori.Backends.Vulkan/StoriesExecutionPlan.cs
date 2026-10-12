namespace Satori.Backends.Vulkan;

public sealed record StoriesExecutionPlan(string Identity,double[] CpuSamplesMs,double[] VulkanSamplesMs,double[]? ResidentSamplesMs=null)
{
    public StoriesVulkanPath? SelectedPath
    {
        get
        {
            double legacy=Median(VulkanSamplesMs),resident=ResidentSamplesMs is null?double.PositiveInfinity:Median(ResidentSamplesMs);
            if(Math.Min(legacy,resident)>=Median(CpuSamplesMs)*0.95)return null;
            return resident<legacy?StoriesVulkanPath.Resident:StoriesVulkanPath.MatVec;
        }
    }
    public string SelectedBackend=>SelectedPath is null?"CPU":"Vulkan";
    public string SelectedKernel=>SelectedPath switch
    {
        StoriesVulkanPath.Resident=>"Q8_0 resident/FP32/tree64-small/row64-output/v1",
        StoriesVulkanPath.MatVec=>"Q8_0 packed/row64/SPIR-V1.3",
        _=>"Q8_0 CPU automatic"
    };
    public bool IsValidFor(string identity)=>Identity==identity&&Valid(CpuSamplesMs)&&Valid(VulkanSamplesMs)&&(ResidentSamplesMs is null||Valid(ResidentSamplesMs));
    private static bool Valid(double[]? samples)=>samples is {Length:5}&&samples.All(v=>double.IsFinite(v)&&v>0);
    private static double Median(double[] samples)=>samples.Order().ElementAt(samples.Length/2);
}

namespace Satori.Backends.Vulkan;

public sealed record StoriesExecutionPlan(string Identity, double[] CpuSamplesMs, double[] VulkanSamplesMs)
{
    public string SelectedBackend => Median(VulkanSamplesMs)<Median(CpuSamplesMs)*0.95?"Vulkan":"CPU";
    public string SelectedKernel => SelectedBackend=="Vulkan"?"Q8_0 packed/row64/SPIR-V1.3":"Q8_0 CPU automatic";
    public bool IsValidFor(string identity) => Identity==identity && Valid(CpuSamplesMs) && Valid(VulkanSamplesMs);
    private static bool Valid(double[]? samples) => samples is {Length:5} && samples.All(v=>double.IsFinite(v) && v>0);
    private static double Median(double[] samples) => samples.Order().ElementAt(samples.Length/2);
}

namespace Satori.Backends.Vulkan;

internal readonly record struct VulkanSubmissionTiming(double PreparationMs,double SubmitMs,double WaitMs,double ValidationMs);

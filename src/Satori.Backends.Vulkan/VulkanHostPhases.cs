namespace Satori.Backends.Vulkan;

/// <summary>Disjoint host intervals; fence wait includes overlapping device execution.</summary>
public readonly record struct VulkanHostPhases(double UploadMs, double RecordingMs, double SubmissionPreparationMs,
    double QueueSubmitMs, double FenceWaitMs, double PostWaitValidationMs, double QueryMs, double ReadbackMs, double OtherMs)
{
    public static VulkanHostPhases operator +(VulkanHostPhases a,VulkanHostPhases b) => new(
        a.UploadMs+b.UploadMs,a.RecordingMs+b.RecordingMs,a.SubmissionPreparationMs+b.SubmissionPreparationMs,
        a.QueueSubmitMs+b.QueueSubmitMs,a.FenceWaitMs+b.FenceWaitMs,a.PostWaitValidationMs+b.PostWaitValidationMs,
        a.QueryMs+b.QueryMs,a.ReadbackMs+b.ReadbackMs,a.OtherMs+b.OtherMs);
}

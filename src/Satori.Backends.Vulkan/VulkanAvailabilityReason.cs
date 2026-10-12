namespace Satori.Backends.Vulkan;

public enum VulkanAvailabilityReason
{
    Ready, LoaderUnavailable, NoDevice, UnsupportedDevice, ValidationUnavailable,
    InitializationFailed, DeviceMemoryUnavailable
}

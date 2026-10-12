namespace Satori.Validation;

public sealed record NumericalResult(
    string Name, int Rows, int Columns, int BlockSize, int PackedBytes,
    double BytesPerWeight, double WeightQuantizationRms,
    double OutputQuantizationRms, double KernelRms, double KernelMaxAbsoluteError,
    double MaxErrorToBoundRatio, bool Passed);

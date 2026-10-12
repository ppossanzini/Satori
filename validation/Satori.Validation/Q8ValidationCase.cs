using Satori.Core;

namespace Satori.Validation;

public sealed record Q8ValidationCase(string Name, QuantizedMatrixShape Shape, byte[] Payload, float[] Input);

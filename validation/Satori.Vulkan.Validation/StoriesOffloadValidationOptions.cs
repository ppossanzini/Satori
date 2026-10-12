internal sealed record StoriesOffloadValidationOptions(bool Validation = true, int Warmups = 3, int Samples = 7, int DecodeTokens = 8, bool MeasureKernelTime = true, bool MeasureHostPhases = false);

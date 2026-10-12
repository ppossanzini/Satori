using Satori.Core;

namespace Satori.Formats.Gguf;

public sealed record StoriesGenerationOptions(int MaxNewTokens = 32, bool Scalar = false, SamplingOptions? Sampling = null);

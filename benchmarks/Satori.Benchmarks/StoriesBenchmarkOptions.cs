namespace Satori.Benchmarks;

internal sealed record StoriesBenchmarkOptions(string ModelPath, bool Scalar, int Warmups = 3, int Samples = 15, int NewTokens = 12);

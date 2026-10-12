namespace Satori.Benchmarks;

internal sealed record MappedStartupResult(
    int Size, int BlockSize, int Trial, double OpenAndValidateMilliseconds,
    double FirstMatVecAfterValidationMicroseconds, bool MatchesOwned);

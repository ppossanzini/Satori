namespace Satori.Validation;

public sealed record Q8Comparison(double MaxAbsoluteError, double RmsError, double MaxErrorOverBound, bool Passed);

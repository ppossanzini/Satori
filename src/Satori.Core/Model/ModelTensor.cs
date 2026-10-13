namespace Satori.Core.Model;

/// <summary>
/// A tensor declared by the container. Dimensions are the declared ones, verbatim, as Int32; a declared
/// dimension outside Int32 is reported as a <c>shape</c> gap rather than truncated.
/// </summary>
public sealed record ModelTensor(
    string Name,
    int FormatId,
    string FormatName,
    bool FormatSupported,
    IReadOnlyList<int> Dimensions,
    long PayloadOffset,
    long PayloadBytes,
    bool ByteLengthKnown)
{
    /// <summary>Product of the declared dimensions, or <c>-1</c> when it overflows Int64.</summary>
    public long ElementCount
    {
        get
        {
            try
            {
                long product = 1;
                for (int i = 0; i < Dimensions.Count; i++)
                    product = checked(product * Dimensions[i]);
                return product;
            }
            catch (OverflowException)
            {
                return -1;
            }
        }
    }
}

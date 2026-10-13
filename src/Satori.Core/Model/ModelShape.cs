namespace Satori.Core.Model;

/// <summary>
/// Dimensions derived from validated metadata. Every value comes from the descriptor; nothing is a nominal
/// constant and <see cref="HeadDim"/> is taken from an explicit key/value length when the metadata declares
/// one, never forced to <c>EmbeddingLength / HeadCount</c>.
/// </summary>
public sealed record ModelShape(
    int Layers,
    int EmbeddingLength,
    int FeedForwardLength,
    int HeadCount,
    int HeadCountKv,
    int HeadDim,
    int VocabularySize,
    bool TiedEmbeddings,
    double RmsNormEpsilon,
    double RopeBase)
{
    /// <summary>Width of a query projection (<c>HeadCount * HeadDim</c>), checked.</summary>
    public int QueryWidth => checked(HeadCount * HeadDim);

    /// <summary>Width of a key/value projection (<c>HeadCountKv * HeadDim</c>), checked.</summary>
    public int KvWidth => checked(HeadCountKv * HeadDim);
}

namespace Satori.Formats.Gguf;

/// <summary>Owned synchronous execution state. Append receives one embedding and commits logits after completion.</summary>
public interface IStoriesSessionExecution : IDisposable
{
    void Append(ReadOnlySpan<float> embedding,int position,Span<float> logits,StoriesStateObserver? observer=null);
    void Reset();
}

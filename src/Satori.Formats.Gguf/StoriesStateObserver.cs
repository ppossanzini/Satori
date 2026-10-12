namespace Satori.Formats.Gguf;

/// <summary>Observes a transient last-token state; copy values to retain them.</summary>
public delegate void StoriesStateObserver(string name, ReadOnlySpan<float> values);

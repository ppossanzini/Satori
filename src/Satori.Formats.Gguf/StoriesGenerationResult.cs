using System.Collections.ObjectModel;

namespace Satori.Formats.Gguf;

public sealed class StoriesGenerationResult
{
    internal StoriesGenerationResult(int promptTokenCount, int[] tokens, string text, StoriesStopReason stopReason)
    {
        PromptTokenCount = promptTokenCount;
        Tokens = Array.AsReadOnly(tokens);
        Text = text;
        StopReason = stopReason;
    }

    public int PromptTokenCount { get; }
    /// <summary>Generated IDs, including EOS when selected.</summary>
    public ReadOnlyCollection<int> Tokens { get; }
    public string Text { get; }
    public StoriesStopReason StopReason { get; }
}

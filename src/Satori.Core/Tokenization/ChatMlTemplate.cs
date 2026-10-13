using System.Text;

namespace Satori.Core.Tokenization;

/// <summary>
/// ChatML conversation template: <c>&lt;|im_start|&gt;role\ncontent&lt;|im_end|&gt;\n</c> for each message.
/// The marker strings and an optional default system message are supplied by the caller from the artifact's
/// special tokens and template, so the renderer stays generic.
/// </summary>
public sealed class ChatMlTemplate(string startToken, string endToken, string? defaultSystem = null)
{
    public string StartToken { get; } = startToken;
    public string EndToken { get; } = endToken;
    public string? DefaultSystem { get; } = defaultSystem;

    public string Render(IReadOnlyList<(string Role, string Content)> messages, bool addGenerationPrompt = false)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var text = new StringBuilder();
        if (DefaultSystem is { Length: > 0 } system && (messages.Count == 0 || messages[0].Role != "system"))
            Append(text, "system", system);
        foreach (var (role, content) in messages)
            Append(text, role, content);
        if (addGenerationPrompt)
        {
            text.Append(StartToken).Append("assistant").Append('\n');
        }
        return text.ToString();
    }

    private void Append(StringBuilder text, string role, string content)
    {
        text.Append(StartToken).Append(role).Append('\n').Append(content).Append(EndToken).Append('\n');
    }
}

using System.Text;
using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>Configurable raw completion with a fresh session for each prompt.</summary>
public sealed class StoriesGenerator : IDisposable
{
    private readonly StoriesModel _model;
    private readonly StoriesTokenizer _tokenizer;
    private readonly object _sync = new();
    private bool _disposed;

    public StoriesGenerator(string path)
    {
        _model = new(path);
        try
        {
            using var file = new GgufFile(path);
            _tokenizer = new(file);
        }
        catch { _model.Dispose(); throw; }
    }

    /// <summary>Emits only continuation text. Callback failures propagate; the session is always released.</summary>
    public StoriesGenerationResult Generate(string prompt, StoriesGenerationOptions? options = null, Action<string>? onText = null)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            options ??= new();
            if (options.MaxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(options),"MaxNewTokens must be nonnegative.");
            var sampler = new TokenSampler(StoriesModel.VocabularySize,options.Sampling);
            int[] promptTokens = _tokenizer.Encode(prompt);
            if (promptTokens.Length > StoriesModel.ContextLength)
                throw new ArgumentException("Prompt exceeds the 128-token context, including BOS.",nameof(prompt));
            int remaining = StoriesModel.ContextLength-promptTokens.Length;
            int limit = Math.Min(options.MaxNewTokens,remaining);
            StoriesStopReason reason = remaining <= options.MaxNewTokens ? StoriesStopReason.ContextLimit : StoriesStopReason.TokenLimit;
            if (limit == 0) return new(promptTokens.Length,[],"",reason);
            using var session = _model.CreateSession(options.Scalar);
            float[] logits = new float[StoriesModel.VocabularySize];
            foreach (int token in promptTokens) _ = session.TryAppend(token,logits);
            var decoder = _tokenizer.CreateDecoder(promptTokens[^1]);
            var text = new StringBuilder();
            var generated = new List<int>(limit);
            void Emit(string part)
            {
                if (part.Length == 0) return;
                text.Append(part);
                onText?.Invoke(part);
            }
            for (int i = 0; i < limit; i++)
            {
                int token = sampler.Select(logits);
                generated.Add(token);
                if (token == StoriesTokenizer.Eos) { reason = StoriesStopReason.Eos; break; }
                Emit(decoder.Append(token));
                if (i+1 < limit && !session.TryAppend(token,logits))
                    throw new InvalidOperationException("Generation exceeded its reserved context.");
            }
            Emit(decoder.Flush());
            return new(promptTokens.Length,generated.ToArray(),text.ToString(),reason);
        }
    }

    public void Dispose()
    {
        lock (_sync) { if (_disposed) return; _disposed = true; _model.Dispose(); }
    }
}

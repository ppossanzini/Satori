using System.Security.Cryptography;
using Satori.Core;

namespace Satori.Formats.Gguf;

/// <summary>Exact Stories15M Q8_0 model. Forward recomputes a prefix without retaining sequence state.</summary>
public sealed class StoriesModel : IDisposable
{
    public const int Width = 288, ContextLength = 128, VocabularySize = 32000;
    private const float Epsilon = 1e-5f;
    private readonly GgufFile _file;
    private readonly StoriesLayer[] _layers;
    private readonly GgufQuantizedMatrix _embedding, _output;
    private readonly GgufFloatVector _outputNorm;
    private readonly object _sync = new();
    private bool _disposed;

    public StoriesModel(string path)
    {
        using (var source = File.OpenRead(path))
        {
            if (source.Length != 26671328 || !SHA256.HashData(source).AsSpan().SequenceEqual(
                Convert.FromHexString("2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6")))
                throw new NotSupportedException("Expected the pinned Stories15M Q8_0 artifact.");
        }
        _file = new GgufFile(path);
        try
        {
            ValidateProfile();
            _embedding = new(_file,"token_embd.weight");
            _output = new(_file,"output.weight");
            _outputNorm = new(_file,"output_norm.weight");
            _layers = Enumerable.Range(0,6).Select(i=>new StoriesLayer(_file,i)).ToArray();
        }
        catch { _file.Dispose(); throw; }
    }

    private void ValidateProfile()
    {
        void Metadata(string key, object expected)
        {
            if (!Equals(_file.ReadScalar(key), expected)) throw new NotSupportedException($"Unsupported {key}.");
        }
        Metadata("general.architecture","llama");
        Metadata("llama.context_length",128u);
        Metadata("llama.embedding_length",288u);
        Metadata("llama.feed_forward_length",768u);
        Metadata("llama.attention.head_count",6u);
        Metadata("llama.block_count",6u);
        Metadata("llama.rope.dimension_count",48u);
        Metadata("llama.attention.layer_norm_rms_epsilon",Epsilon);
        if (_file.Tensors.Count != 57) throw new NotSupportedException("Expected 57 tensors.");
        void Tensor(string name, uint type, params long[] dimensions)
        {
            if (!_file.Tensors.TryGetValue(name,out var tensor) || tensor.Type != type || !tensor.Dimensions.SequenceEqual(dimensions))
                throw new NotSupportedException($"Unsupported tensor {name}.");
        }
        Tensor("token_embd.weight",8,288,32000); Tensor("output.weight",8,288,32000); Tensor("output_norm.weight",0,288);
        for (int i = 0; i < 6; i++)
        {
            string prefix = $"blk.{i}.";
            Tensor(prefix+"attn_norm.weight",0,288); Tensor(prefix+"ffn_norm.weight",0,288);
            foreach (string name in new[] {"attn_q","attn_k","attn_v","attn_output"}) Tensor(prefix+name+".weight",8,288,288);
            Tensor(prefix+"ffn_gate.weight",8,288,768); Tensor(prefix+"ffn_up.weight",8,288,768); Tensor(prefix+"ffn_down.weight",8,768,288);
        }
    }

    /// <summary>Writes last-token logits. Prefix is nonempty and at most 128 IDs.
    /// Calls on one model are serialized. Scalar selects the verified scalar MatVec.</summary>
    public void Forward(ReadOnlySpan<int> tokens, Span<float> logits, bool scalar = false, StoriesStateObserver? observer = null)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            if (tokens.IsEmpty || tokens.Length > ContextLength) throw new ArgumentOutOfRangeException(nameof(tokens));
            if (logits.Length != VocabularySize) throw new ArgumentException("Expected 32000 logits.",nameof(logits));
            foreach (int token in tokens) if ((uint)token >= VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokens));
            if (System.Runtime.InteropServices.MemoryMarshal.AsBytes(tokens).Overlaps(System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits)))
                throw new ArgumentException("Token input overlaps logits.");
            float[] keys = new float[6*tokens.Length*Width], values = new float[keys.Length];
            float[] x = new float[Width], norm = new float[Width], query = new float[Width], attention = new float[Width], projected = new float[Width];
            float[] gate = new float[768], up = new float[768];
            double[] scores = new double[tokens.Length];
            void MatVec(GgufQuantizedMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
            {
                if (scalar) matrix.ExecuteScalar(input,output); else matrix.Execute(input,output);
            }
            for (int position = 0; position < tokens.Length; position++)
            {
                bool capture = position == tokens.Length-1 && observer is not null;
                _embedding.ReadRow(tokens[position],x);
                for (int layerIndex = 0; layerIndex < 6; layerIndex++)
                {
                    var layer = _layers[layerIndex];
                    int start = layerIndex*tokens.Length*Width;
                    Span<float> key = keys.AsSpan(start+position*Width,Width), value = values.AsSpan(start+position*Width,Width);
                    TransformerOps.RmsNorm(x,layer.AttentionNorm.Values,Epsilon,norm);
                    if (capture && layerIndex == 0) observer!("attn_norm-0",norm);
                    MatVec(layer.Query,norm,query); MatVec(layer.Key,norm,key); MatVec(layer.Value,norm,value);
                    TransformerOps.Rope(query,48,position); TransformerOps.Rope(key,48,position);
                    if (capture && layerIndex == 0) { observer!("Qcur-0",query); observer!("Kcur-0",key); observer!("Vcur-0",value); }
                    TransformerOps.Attention(query,keys.AsSpan(start,(position+1)*Width),values.AsSpan(start,(position+1)*Width),48,position+1,scores,attention);
                    MatVec(layer.AttentionOutput,attention,projected);
                    if (capture && layerIndex == 0) observer!("attn_out-0",projected);
                    TransformerOps.Add(x,projected,x);
                    if (capture && layerIndex == 0) observer!("ffn_inp-0",x);
                    TransformerOps.RmsNorm(x,layer.FeedForwardNorm.Values,Epsilon,norm);
                    if (capture && layerIndex == 0) observer!("ffn_norm-0",norm);
                    MatVec(layer.Gate,norm,gate); MatVec(layer.Up,norm,up);
                    TransformerOps.SwiGlu(gate,up,gate);
                    MatVec(layer.Down,gate,projected);
                    if (capture && layerIndex == 0) observer!("ffn_out-0",projected);
                    TransformerOps.Add(x,projected,x);
                    if (capture) observer!(layer.OutputState,x);
                }
                if (position == tokens.Length-1)
                {
                    TransformerOps.RmsNorm(x,_outputNorm.Values,Epsilon,norm);
                    if (capture) observer!("result_norm",norm);
                    MatVec(_output,norm,logits);
                }
            }
        }
    }

    public StoriesSession CreateSession(bool scalar = false)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            return new(this,scalar);
        }
    }

    internal void ComputeNext(StoriesWorkspace workspace, int token, int position, Span<float> logits, bool scalar)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed,this);
            void MatVec(GgufQuantizedMatrix matrix, ReadOnlySpan<float> input, Span<float> output)
            {
                if (scalar) matrix.ExecuteScalar(input,output); else matrix.Execute(input,output);
            }
            var x = workspace.Hidden;
            var norm = workspace.Norm;
            var projected = workspace.Projected;
            _embedding.ReadRow(token,x);
            for (int layerIndex = 0; layerIndex < 6; layerIndex++)
            {
                var layer = _layers[layerIndex];
                int start = layerIndex*ContextLength*Width;
                Span<float> key = workspace.Keys.AsSpan(start+position*Width,Width);
                Span<float> value = workspace.Values.AsSpan(start+position*Width,Width);
                TransformerOps.RmsNorm(x,layer.AttentionNorm.Values,Epsilon,norm);
                MatVec(layer.Query,norm,workspace.Query); MatVec(layer.Key,norm,key); MatVec(layer.Value,norm,value);
                TransformerOps.Rope(workspace.Query,48,position); TransformerOps.Rope(key,48,position);
                TransformerOps.Attention(workspace.Query,workspace.Keys.AsSpan(start,(position+1)*Width),
                    workspace.Values.AsSpan(start,(position+1)*Width),48,position+1,workspace.Scores,workspace.Attention);
                MatVec(layer.AttentionOutput,workspace.Attention,projected);
                TransformerOps.Add(x,projected,x);
                TransformerOps.RmsNorm(x,layer.FeedForwardNorm.Values,Epsilon,norm);
                MatVec(layer.Gate,norm,workspace.Gate); MatVec(layer.Up,norm,workspace.Up);
                TransformerOps.SwiGlu(workspace.Gate,workspace.Up,workspace.Gate);
                MatVec(layer.Down,workspace.Gate,projected);
                TransformerOps.Add(x,projected,x);
            }
            TransformerOps.RmsNorm(x,_outputNorm.Values,Epsilon,norm);
            MatVec(_output,norm,logits);
        }
    }

    public void Dispose()
    {
        lock (_sync) { if (_disposed) return; _disposed = true; _file.Dispose(); }
    }
}

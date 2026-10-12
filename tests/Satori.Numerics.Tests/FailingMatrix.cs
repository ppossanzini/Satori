using Satori.Core;

namespace Satori.Numerics.Tests;

internal sealed class FailingMatrix(byte[] payload, QuantizedMatrixShape shape, Action beforeExecute, Action onDispose) : IQuantizedMatrix
{
    private bool _disposed;
    public void Execute(ReadOnlySpan<float> input,Span<float> output) { beforeExecute(); QuantizedMatVec.Execute(payload,shape,input,output); }
    public void Dispose() { if(_disposed) return; _disposed=true;onDispose(); }
}

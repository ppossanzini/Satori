using Satori.Core;

namespace Satori.Numerics.Tests;

internal sealed class FailingMatrixFactory(int failAfter=int.MaxValue) : IQuantizedMatrixFactory
{
    private int _created;
    private int _calls;
    internal int CallsBeforeFailure { get; set; }=10;
    internal bool FailExecution { get; set; }=true;
    internal int Disposed { get; private set; }
    public IQuantizedMatrix Create(ReadOnlySpan<byte> payload,QuantizedMatrixShape shape)
    {
        if(_created++==failAfter) throw new IOException("Resident allocation failed.");
        return new FailingMatrix(payload.ToArray(),shape,()=> { if(++_calls>=CallsBeforeFailure && FailExecution) throw new IOException("Execution failed after partial cache updates."); },()=>Disposed++);
    }
}

using System.Buffers.Binary;
using Satori.Core;

namespace Satori.Validation;

/// <summary>Independent FP64 decode/dot oracle over serialized GGUF Q8_0 blocks.</summary>
public static class Q8Validation
{
    public static IEnumerable<Q8ValidationCase> CriticalCases()
    {
        foreach(int rows in new[] {1,7,63,64,65,129})
        foreach(int columns in new[] {32,64,96,288,768})
        {
            var shape=new QuantizedMatrixShape(rows,columns,BlockQuantization.Q8_0);
            byte[] payload=new byte[shape.ByteLength];
            float[] scales=[0.5f,-2f,0f,-0f,(float)Half.Epsilon,65504f,1f];
            for(int row=0;row<rows;row++)
            for(int block=0;block<columns/32;block++)
            {
                int offset=(row*(columns/32)+block)*34;
                BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset),BitConverter.HalfToUInt16Bits((Half)scales[(row+block)%scales.Length]));
                for(int j=0;j<32;j++) payload[offset+2+j]=unchecked((byte)(sbyte)((row*32+block*17+j*7)%256-128));
            }
            yield return new($"signed-scales-{rows}x{columns}",shape,payload,Input(columns));
        }
    }

    public static float[] Input(int columns) => Enumerable.Range(0,columns).Select(i=>(i*17%31-15)/16f).ToArray();

    public static Q8Comparison Compare(ReadOnlySpan<byte> payload, QuantizedMatrixShape shape, ReadOnlySpan<float> input, ReadOnlySpan<float> actual)
    {
        double max=0,squared=0,ratio=0;
        bool passed=true;
        double n=shape.Columns+shape.Columns/32.0+2;
        double gamma=n*Math.ScaleB(1.0,-24)/(1-n*Math.ScaleB(1.0,-24));
        for(int row=0;row<shape.Rows;row++)
        {
            double expected=0,magnitude=0;
            for(int col=0;col<shape.Columns;col++)
            {
                int offset=(row*(shape.Columns/32)+col/32)*34;
                double scale=(double)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(offset,2)));
                double product=scale*(sbyte)payload[offset+2+col%32]*input[col];
                expected+=product; magnitude+=Math.Abs(product);
            }
            double bound=gamma*magnitude+n*float.Epsilon;
            double error=Math.Abs(actual[row]-expected);
            max=Math.Max(max,error); squared+=error*error;
            ratio=Math.Max(ratio,error/Math.Max(bound,double.Epsilon));
            passed&=double.IsFinite(error) && error<=bound;
        }
        return new(max,Math.Sqrt(squared/shape.Rows),ratio,passed);
    }
}

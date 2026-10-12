using Satori.Core;

namespace Satori.Validation;

public static class NumericalValidation
{
    public static NumericalResult Evaluate(NumericalCase sample, int blockSize, bool useAvx2 = false)
    {
        var matrix = Ternary2Matrix.Quantize(sample.Weights, sample.Rows, sample.Columns, blockSize);
        float[] reconstructed = new float[sample.Weights.Length];
        matrix.Dequantize(reconstructed);
        float[] output = new float[sample.Rows];
        if (useAvx2) Ternary2MatVec.ExecuteAvx2(matrix, sample.Input, output);
        else Ternary2MatVec.ExecuteScalar(matrix, sample.Input, output);
        double weightSquared = 0, quantizationSquared = 0, kernelSquared = 0, maxError = 0, maxRatio = 0;
        bool passed = true;
        for (int i = 0; i < reconstructed.Length; i++)
        {
            double difference = (double)reconstructed[i] - sample.Weights[i];
            weightSquared += difference * difference;
        }

        int blocks = (sample.Columns + blockSize - 1) / blockSize;
        double operations = blockSize + blocks + 2;
        double epsilon = Math.ScaleB(1d, -24);
        double gamma = operations * epsilon / (1 - operations * epsilon);
        for (int row = 0; row < sample.Rows; row++)
        {
            double original = 0, quantized = 0, absoluteSum = 0;
            for (int column = 0; column < sample.Columns; column++)
            {
                int index = row * sample.Columns + column;
                original += (double)sample.Weights[index] * sample.Input[column];
                double term = (double)reconstructed[index] * sample.Input[column];
                quantized += term;
                absoluteSum += Math.Abs(term);
            }
            double quantizationError = quantized - original;
            double kernelError = Math.Abs(output[row] - quantized);
            double bound = gamma * absoluteSum + (sample.Columns + blocks) * (double)float.Epsilon;
            double ratio = kernelError / bound;
            quantizationSquared += quantizationError * quantizationError;
            kernelSquared += kernelError * kernelError;
            maxError = Math.Max(maxError, kernelError);
            maxRatio = Math.Max(maxRatio, ratio);
            passed &= float.IsFinite(output[row]) && double.IsFinite(ratio) && ratio <= 1;
        }
        return new(sample.Name, sample.Rows, sample.Columns, blockSize, matrix.ByteLength,
            (double)matrix.ByteLength / reconstructed.Length,
            Math.Sqrt(weightSquared / reconstructed.Length), Math.Sqrt(quantizationSquared / sample.Rows),
            Math.Sqrt(kernelSquared / sample.Rows), maxError, maxRatio, passed);
    }
}

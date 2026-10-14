using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Satori.Core;

public static class QuantizedMatVec
{
    /// <summary>Selects AVX2 when available, otherwise scalar, on validated immutable weights.</summary>
    public static void Execute(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (Avx2.IsSupported) ExecuteAvx2(data, shape, input, output);
        else ExecuteScalar(data, shape, input, output);
    }

    public static void ExecuteAvx2(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("AVX2 is unavailable.");
        Guard(data, shape, input, output);
        if (shape.Format is BlockQuantization.Q4_K or BlockQuantization.Q6_K)
            QuantizedKAvx2Kernel.Execute(data, shape, input, output);
        else
            QuantizedAvx2Kernel.Execute(data, shape, input, output);
    }

    public static void ValidatePayload(ReadOnlySpan<byte> data, QuantizedMatrixShape shape)
    {
        if (data.Length != shape.ByteLength) throw new ArgumentException("Incorrect quantized payload length.", nameof(data));
        for (int offset = 0; offset < data.Length; offset += shape.BlockBytes)
        {
            var block = data[offset..];
            bool finite = shape.Format switch
            {
                BlockQuantization.Q4_K => Q4KCodec.AreScalesFinite(block),
                BlockQuantization.Q6_K => Q6KCodec.AreScalesFinite(block),
                _ => float.IsFinite(Scale(block)),
            };
            if (!finite) throw new InvalidDataException($"Nonfinite block scale at {offset}.");
        }
    }

    /// <summary>Multiplies immutable compressed weights whose scales have already been validated.</summary>
    public static void ExecuteScalar(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, Span<float> output)
    {
        Guard(data, shape, input, output);
        for (int row = 0; row < shape.Rows; row++)
        {
            float total = 0;
            for (int column = 0; column < shape.Columns; column += shape.BlockElements)
            {
                int offset = (row * (shape.Columns / shape.BlockElements) + column / shape.BlockElements) * shape.BlockBytes;
                var block = data.Slice(offset, shape.BlockBytes);
                if (shape.Format == BlockQuantization.Q4_K)
                {
                    total += Q4KCodec.DotBlock(block, input.Slice(column, Q4KCodec.ElementsPerBlock));
                    continue;
                }
                if (shape.Format == BlockQuantization.Q6_K)
                {
                    total += Q6KCodec.DotBlock(block, input.Slice(column, Q6KCodec.ElementsPerBlock));
                    continue;
                }
                float sum = 0;
                if (shape.Format == BlockQuantization.Q8_0)
                    for (int j = 0; j < 32; j++) sum += (sbyte)block[2 + j] * input[column + j];
                else
                {
                    for (int j = 0; j < 16; j++) sum += ((block[2 + j] & 15) - 8) * input[column + j];
                    for (int j = 0; j < 16; j++) sum += ((block[2 + j] >> 4) - 8) * input[column + 16 + j];
                }
                total += Scale(block) * sum;
            }
            output[row] = total;
        }
    }

    internal static void Guard(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input, Span<float> output)
    {
        if (data.Length != shape.ByteLength || input.Length != shape.Columns || output.Length != shape.Rows)
            throw new ArgumentException("Matrix/vector dimensions do not match.");
        if (input.Overlaps(output)) throw new ArgumentException("Input and output must not overlap.");
    }

    /// <summary>
    /// Multiplies compressed weights by several input vectors in one pass. <paramref name="input"/> holds
    /// <paramref name="positions"/> vectors of <c>Columns</c> floats laid out position after position, and
    /// <paramref name="output"/> receives the same number of vectors of <c>Rows</c> floats in the same layout.
    /// Each weight block is decoded once and then dotted with every position, so the packed bytes are read
    /// once for the whole call rather than once per position: that reuse, not the arithmetic, is what makes a
    /// prompt affordable. Work is split across <paramref name="threads"/> workers, and no worker writes a row
    /// another worker writes, so the result does not depend on the split.
    /// </summary>
    /// <remarks>
    /// The inner dot uses fused multiply-add when the host has it, which rounds once per product instead of
    /// twice. The batched result can therefore differ from <see cref="Execute"/> in the last bits while
    /// agreeing to within the tolerance the tests assert; both compute the same mathematical quantity.
    /// </remarks>
    public static unsafe void ExecuteBatch(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input,
        int positions, Span<float> output, int threads)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(positions);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(threads);
        if (data.Length != shape.ByteLength) throw new ArgumentException("Incorrect quantized payload length.", nameof(data));
        if (input.Length != Checked(positions, shape.Columns))
            throw new ArgumentException("Input must hold one vector per position.", nameof(input));
        if (output.Length != Checked(positions, shape.Rows))
            throw new ArgumentException("Output must hold one vector per position.", nameof(output));
        if (MemoryMarshal.AsBytes(input).Overlaps(MemoryMarshal.AsBytes(output)))
            throw new ArgumentException("Input and output must not overlap.");
        if (Avx2.IsSupported && shape.Format is BlockQuantization.Q4_K or BlockQuantization.Q6_K)
            ExecuteBatchQuantized(data, shape, input, positions, output, threads);
        else
            ExecuteBatchFloat(data, shape, input, positions, output, threads);
    }

    /// <summary>
    /// Integer batched projection. The activation vectors are quantized to Q8_K once for the whole call and
    /// then reused by every row, which is what makes the integer kernel pay: the activation is read once per
    /// matrix instead of once per row, and each product is an integer multiply-add.
    /// </summary>
    /// <remarks>
    /// Quantization is lossy by construction, so this path differs from <see cref="Execute"/> in the last bits
    /// and on low-order terms; the tests bound that difference rather than hiding it. An all-zero activation
    /// block produces a zero scale and the kernel skips it, so no code is multiplied by a meaningless one.
    /// The rows are claimed by the workers from a shared counter in chunks of <see cref="ChunksPerWorker"/>
    /// each, so a worker that draws slower cores takes fewer of them instead of holding the whole call at the
    /// barrier; chunks are claimed once each, so the split still cannot change the result.
    /// <paramref name="positions"/> is bounded by the kernel's accumulator size.
    /// </remarks>
    private static unsafe void ExecuteBatchQuantized(ReadOnlySpan<byte> data, QuantizedMatrixShape shape,
        ReadOnlySpan<float> input, int positions, Span<float> output, int threads)
    {
        int rowBytes = Q8KCodec.RowBytes(shape.Columns);
        byte[] quantized = ArrayPool<byte>.Shared.Rent(Checked(positions, rowBytes));
        try
        {
            for (int position = 0; position < positions; position++)
                Q8KCodec.QuantizeRow(input.Slice(position * shape.Columns, shape.Columns),
                    quantized.AsSpan(position * rowBytes, rowBytes));

            fixed (byte* dataPointer = &MemoryMarshal.GetReference(data))
            fixed (byte* activationPointer = &quantized[0])
            fixed (float* outputPointer = &MemoryMarshal.GetReference(output))
            {
                int workers = Math.Min(threads, shape.Rows);
                if (workers <= 1)
                {
                    QuantizedKAvx2Kernel.ExecuteQuantizedTiled(dataPointer, shape, activationPointer, positions,
                        outputPointer, 0, shape.Rows);
                    return;
                }
                // Rows are claimed from a shared counter rather than dealt out in equal parts. An equal split
                // makes the call last as long as its slowest worker, and on a processor with cores of
                // different speeds the workers do not finish together, so every call waits for whichever
                // worker drew the slow cores. The claim costs one atomic per chunk and the chunks stay
                // disjoint and complete, so the result is the one an equal split produces.
                int chunk = Math.Max(1, (shape.Rows + workers * ChunksPerWorker - 1) / (workers * ChunksPerWorker));
                int[] claimed = new int[1];
                var running = new Task[workers];
                for (int worker = 0; worker < workers; worker++)
                {
                    byte* workerData = dataPointer;
                    byte* workerActivations = activationPointer;
                    float* workerOutput = outputPointer;
                    running[worker] = Task.Factory.StartNew(() =>
                    {
                        while (true)
                        {
                            int begin = Interlocked.Add(ref claimed[0], chunk) - chunk;
                            if (begin >= shape.Rows) return;
                            QuantizedKAvx2Kernel.ExecuteQuantizedTiled(workerData, shape, workerActivations, positions,
                                workerOutput, begin, Math.Min(begin + chunk, shape.Rows));
                        }
                    }, CancellationToken.None, TaskCreationOptions.DenyChildAttach, TaskScheduler.Default);
                }
                Task.WaitAll(running);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(quantized);
        }
    }

    private static unsafe void ExecuteBatchFloat(ReadOnlySpan<byte> data, QuantizedMatrixShape shape, ReadOnlySpan<float> input,
        int positions, Span<float> output, int threads)
    {
        int workers = Math.Min(threads, shape.Rows);
        if (workers <= 1)
        {
            float[] workspace = new float[WorkspacePerWorker(shape, positions)];
            fixed (byte* dataPointer = &MemoryMarshal.GetReference(data))
            fixed (float* inputPointer = &MemoryMarshal.GetReference(input))
            fixed (float* outputPointer = &MemoryMarshal.GetReference(output))
                ExecuteBatchRows(dataPointer, shape, inputPointer, positions, outputPointer, 0, shape.Rows, 0, workspace);
            return;
        }

        // A worker needs the spans as raw addresses: a ref struct cannot be captured by the task body, while a
        // pointer can. The weights are immutable and the row ranges are disjoint, so nothing is shared.
        fixed (byte* dataPointer = &MemoryMarshal.GetReference(data))
        fixed (float* inputPointer = &MemoryMarshal.GetReference(input))
        fixed (float* outputPointer = &MemoryMarshal.GetReference(output))
        {
            float[] workspace = ArrayPool<float>.Shared.Rent(Checked(WorkspacePerWorker(shape, positions), workers));
            try
            {
                int stride = (shape.Rows + workers - 1) / workers;
                var running = new Task[workers];
                for (int worker = 0; worker < workers; worker++)
                {
                    int begin = worker * stride;
                    int end = Math.Min(begin + stride, shape.Rows);
                    if (begin >= end)
                    {
                        running[worker] = Task.CompletedTask;
                        continue;
                    }
                    byte* workerData = dataPointer;
                    float* workerInput = inputPointer;
                    float* workerOutput = outputPointer;
                    int offset = worker * WorkspacePerWorker(shape, positions);
                    running[worker] = Task.Factory.StartNew(() =>
                        ExecuteBatchRows(workerData, shape, workerInput, positions, workerOutput, begin, end, offset,
                            workspace), CancellationToken.None, TaskCreationOptions.DenyChildAttach, TaskScheduler.Default);
                }
                Task.WaitAll(running);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(workspace);
            }
        }
    }

    /// <summary>Rows of weights one tile holds, so the decoded blocks stay in the first-level cache.</summary>
    private const int RowTile = 16;

    /// <summary>
    /// Chunks of rows each worker may claim when the rows are claimed dynamically. Four per worker keeps the
    /// claim frequency negligible at one atomic per chunk while leaving enough chunks that a worker which drew
    /// slow cores does not hold the whole call.
    /// </summary>
    private const int ChunksPerWorker = 4;

    /// <summary>Floats one worker needs: the weight tile plus one accumulator tile.</summary>
    private static int WorkspacePerWorker(QuantizedMatrixShape shape, int positions) =>
        checked(RowTile * shape.BlockElements + RowTile * positions);

    /// <summary>
    /// One worker's share of <see cref="ExecuteBatch"/>: the rows <c>[begin, end)</c>. Rows are processed in
    /// tiles of <see cref="RowTile"/>, and inside a tile the block loop is outside the position loop, so an
    /// input window is read once and then dotted with every row of the tile. That order is what keeps the
    /// input from being re-read once per row, which is the difference between a prompt that fits the memory
    /// system and one that streams the same bytes thousands of times.
    /// </summary>
    private static unsafe void ExecuteBatchRows(byte* data, QuantizedMatrixShape shape, float* input,
        int positions, float* output, int begin, int end, int workspaceOffset, float[] workspaceBuffer)
    {
        int columns = shape.Columns;
        int blocksPerRow = columns / shape.BlockElements;
        int blockElements = shape.BlockElements;
        Span<float> weights = workspaceBuffer.AsSpan(workspaceOffset, RowTile * blockElements);
        Span<float> accumulator = workspaceBuffer.AsSpan(workspaceOffset + RowTile * blockElements, RowTile * positions);
        for (int tile = begin; tile < end; tile += RowTile)
        {
            int tileRows = Math.Min(RowTile, end - tile);
            accumulator[..checked(tileRows * positions)].Clear();
            for (int block = 0; block < blocksPerRow; block++)
            {
                for (int row = 0; row < tileRows; row++)
                {
                    var packed = new ReadOnlySpan<byte>(
                        data + ((tile + row) * blocksPerRow + block) * shape.BlockBytes, shape.BlockBytes);
                    QuantizedRow.DecodeBlock(packed, shape.Format, weights.Slice(row * blockElements, blockElements));
                }
                int columnBase = block * blockElements;
                for (int position = 0; position < positions; position++)
                {
                    var window = new ReadOnlySpan<float>(input + position * columns + columnBase, blockElements);
                    for (int row = 0; row < tileRows; row++)
                    {
                        int index = row * positions + position;
                        accumulator[index] += Dot(weights.Slice(row * blockElements, blockElements), window);
                    }
                }
            }
            for (int position = 0; position < positions; position++)
                for (int row = 0; row < tileRows; row++)
                    output[position * shape.Rows + tile + row] = accumulator[row * positions + position];
        }
    }

    /// <summary>Dot product of one decoded block with one input window, fused when the host supports it.</summary>
    private static float Dot(ReadOnlySpan<float> weights, ReadOnlySpan<float> input)
    {
        ref float left = ref MemoryMarshal.GetReference(weights);
        ref float right = ref MemoryMarshal.GetReference(input);
        float total = 0;
        int j = 0;
        if (Fma.IsSupported)
        {
            var sum = Vector256<float>.Zero;
            for (; j + Vector256<float>.Count <= weights.Length; j += Vector256<float>.Count)
                sum = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref left, (nuint)j), Vector256.LoadUnsafe(ref right, (nuint)j), sum);
            total = QuantizedKAvx2Kernel.HorizontalSum(sum);
        }
        for (; j < weights.Length; j++) total += weights[j] * input[j];
        return total;
    }

    private static int Checked(int positions, int width) => checked(positions * width);

    internal static float Scale(ReadOnlySpan<byte> block) => (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(block));
}

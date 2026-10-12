using Satori.Core;

namespace Satori.Numerics.Tests;

public class MappedWeightsTests
{
    [Fact]
    public void ReadsBeyondTwoGiBWithoutTruncatingOffsets()
    {
        long offset = (long)int.MaxValue + 123;
        byte[] expected = Enumerable.Range(0, 257).Select(i => (byte)i).ToArray();
        using var fixture = new MappedFileFixture(offset + expected.Length, offset, expected);
        using var weights = new MappedWeights(fixture.Path);
        Assert.Equal(offset + expected.Length, weights.Length);
        weights.Read(offset, expected.Length, static (bytes, state) => Assert.Equal(state, bytes.ToArray()), expected);
    }

    [Fact]
    public void ChecksBoundsBeforeExposingPointerIncludingUnalignedAndEmptySlices()
    {
        byte[] data = Enumerable.Range(0, 16).Select(i => (byte)i).ToArray();
        using var fixture = new MappedFileFixture(data.Length, 0, data);
        using var weights = new MappedWeights(fixture.Path);
        weights.Read(3, 5, static (bytes, _) => Assert.Equal(new byte[] { 3, 4, 5, 6, 7 }, bytes.ToArray()), 0);
        weights.Read(16, 0, static (bytes, _) => Assert.True(bytes.IsEmpty), 0);
        foreach (var (offset, length) in new (long, int)[] { (-1, 1), (0, -1), (15, 2), (17, 0), (long.MaxValue, int.MaxValue) })
            Assert.Throws<ArgumentOutOfRangeException>(() => weights.Read(offset, length, static (_, _) => Assert.Fail("Invalid pointer exposed."), 0));
    }

    [Fact]
    public async Task DisposePreservesConcurrentActiveReadersAndRejectsNewReads()
    {
        byte[] expected = Enumerable.Range(0, 32).Select(i => (byte)(i * 3)).ToArray();
        using var fixture = new MappedFileFixture(expected.Length, 0, expected);
        using var weights = new MappedWeights(fixture.Path);
        using var entered = new CountdownEvent(2);
        using var release = new ManualResetEventSlim();
        Task[] readers = Enumerable.Range(0, 2).Select(_ => Task.Run(() =>
            weights.Read(0, expected.Length, (bytes, _) =>
            {
                entered.Signal();
                if (!release.Wait(TimeSpan.FromSeconds(10)))
                    throw new TimeoutException("Reader was not released.");
                Assert.Equal(expected, bytes.ToArray());
            }, 0))).ToArray();
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)), "Readers did not enter concurrently.");
            weights.Dispose();
            weights.Dispose();
            Assert.Throws<ObjectDisposedException>(() => weights.Read(0, 1, static (_, _) => { }, 0));
        }
        finally
        {
            release.Set();
            await Task.WhenAll(readers).WaitAsync(TimeSpan.FromSeconds(10));
        }
    }

    [Fact]
    public void CallbackMayDisposeOwnerWithoutInvalidatingItsBytes()
    {
        byte[] expected = [1, 2, 3, 4];
        using var fixture = new MappedFileFixture(expected.Length, 0, expected);
        using var weights = new MappedWeights(fixture.Path);
        weights.Read(0, expected.Length, (bytes, _) =>
        {
            weights.Dispose();
            Assert.Equal(expected, bytes.ToArray());
        }, 0);
        Assert.Throws<ObjectDisposedException>(() => weights.Read(0, 1, static (_, _) => { }, 0));
    }

    [Fact]
    public void CallbackExceptionReleasesPointerAndPreservesSubsequentAccess()
    {
        using var fixture = new MappedFileFixture(4, 0, [5, 6, 7, 8]);
        using var weights = new MappedWeights(fixture.Path);
        Assert.Throws<InvalidOperationException>(() => weights.Read(0, 4, static (_, _) => throw new InvalidOperationException("callback"), 0));
        weights.Read(0, 4, static (bytes, _) => Assert.Equal(new byte[] { 5, 6, 7, 8 }, bytes.ToArray()), 0);
    }

    [Fact]
    public void StaticCallbackReadAllocatesNoManagedMemoryAfterWarmup()
    {
        using var fixture = new MappedFileFixture(4, 0, [5, 6, 7, 8]);
        using var weights = new MappedWeights(fixture.Path);
        System.Buffers.ReadOnlySpanAction<byte, int> action = static (bytes, _) => Consume(bytes);
        for (int i = 0; i < 100; i++) weights.Read(0, 4, action, 0);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) weights.Read(0, 4, action, 0);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    private static void Consume(ReadOnlySpan<byte> bytes)
    {
        if (bytes[0] != 5 || bytes[3] != 8)
            throw new InvalidOperationException("Mapped data changed.");
    }
}

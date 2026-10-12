using System.Buffers;
using System.IO.MemoryMappedFiles;
using Microsoft.Win32.SafeHandles;

namespace Satori.Core;

/// <summary>Read-only mapped weights with callback-scoped access and deferred unmapping.</summary>
public sealed class MappedWeights : IDisposable
{
    private readonly object _gate = new();
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly long _pointerOffset;
    private bool _disposed;

    public MappedWeights(string path)
    {
        if (!Environment.Is64BitProcess)
            throw new PlatformNotSupportedException("Weight mappings require a 64-bit process.");

        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        MemoryMappedFile? file = null;
        try
        {
            Length = stream.Length;
            if (Length == 0)
                throw new ArgumentException("Cannot map an empty weight file.", nameof(path));
            file = MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.Read,
                HandleInheritability.None, leaveOpen: false);
            _view = file.CreateViewAccessor(0, Length, MemoryMappedFileAccess.Read);
            _pointerOffset = _view.PointerOffset;
            _file = file;
        }
        catch
        {
            file?.Dispose();
            stream.Dispose();
            throw;
        }
    }

    public long Length { get; }

    /// <summary>Executes an action while the selected bytes remain mapped, without copying them.</summary>
    /// <remarks>Concurrent reads are permitted. Disposal rejects new reads but preserves active callbacks.</remarks>
    public unsafe void Read<TState>(long offset, int length, ReadOnlySpanAction<byte, TState> action, TState state)
        where TState : allows ref struct
    {
        ArgumentNullException.ThrowIfNull(action);
        SafeMemoryMappedViewHandle handle;
        byte* pointer = null;
        nint displacement;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (offset < 0 || length < 0 || offset > Length || length > Length - offset)
                throw new ArgumentOutOfRangeException(nameof(offset), "Read must remain inside the mapped file.");
            displacement = checked((nint)checked(_pointerOffset + offset));
            handle = _view.SafeMemoryMappedViewHandle;
            handle.AcquirePointer(ref pointer);
        }
        try
        {
            action(new ReadOnlySpan<byte>(pointer + displacement, length), state);
        }
        finally
        {
            handle.ReleasePointer();
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
            try
            {
                _view.Dispose();
            }
            finally
            {
                _file.Dispose();
            }
        }
    }
}

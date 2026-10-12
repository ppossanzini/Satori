namespace Satori.Numerics.Tests;

internal sealed class MappedFileFixture : IDisposable
{
    private readonly string _directory;
    public string Path { get; }

    public MappedFileFixture(long length, long offset, byte[] data)
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(System.IO.Path.Combine(root.FullName, "Satori.sln")))
            root = root.Parent;
        if (root is null)
            throw new InvalidOperationException("Cannot locate project temporary workspace.");
        _directory = System.IO.Path.Combine(root.FullName, ".temp", "t09", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        Path = System.IO.Path.Combine(_directory, "weights.bin");
        try
        {
            using var file = new FileStream(Path, FileMode.CreateNew, FileAccess.Write);
            file.SetLength(length);
            file.Position = offset;
            file.Write(data);
        }
        catch
        {
            Directory.Delete(_directory, recursive: true);
            throw;
        }
    }

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}

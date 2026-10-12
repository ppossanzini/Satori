namespace Satori.Formats.Gguf;

/// <summary>Creates independent sequence state; the owner must outlive models borrowing the factory.</summary>
public interface IStoriesSessionExecutionFactory
{
    IStoriesSessionExecution CreateSession();
}

namespace Satori.Formats.Gguf;

internal sealed class StoriesWorkspace
{
    internal float[] Keys { get; } = new float[6*StoriesModel.ContextLength*StoriesModel.Width];
    internal float[] Values { get; } = new float[6*StoriesModel.ContextLength*StoriesModel.Width];
    internal float[] Hidden { get; } = new float[StoriesModel.Width];
    internal float[] Norm { get; } = new float[StoriesModel.Width];
    internal float[] Query { get; } = new float[StoriesModel.Width];
    internal float[] Attention { get; } = new float[StoriesModel.Width];
    internal float[] Projected { get; } = new float[StoriesModel.Width];
    internal float[] Gate { get; } = new float[768];
    internal float[] Up { get; } = new float[768];
    internal double[] Scores { get; } = new double[StoriesModel.ContextLength];

    internal void Clear()
    {
        Array.Clear(Keys); Array.Clear(Values); Array.Clear(Hidden); Array.Clear(Norm);
        Array.Clear(Query); Array.Clear(Attention); Array.Clear(Projected);
        Array.Clear(Gate); Array.Clear(Up); Array.Clear(Scores);
    }
}

namespace OrcaHello.Web.UI.Services
{
    public interface IHydrophoneViewService
    {
        ValueTask<List<HydrophoneItemView>> RetrieveAllHydrophoneViewsAsync();
    }
}

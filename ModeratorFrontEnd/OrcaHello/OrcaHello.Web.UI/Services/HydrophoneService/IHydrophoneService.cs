namespace OrcaHello.Web.UI.Services
{
    public interface IHydrophoneService
    {
        ValueTask<List<Hydrophone>> RetrieveAllHydrophonesAsync();
    }
}

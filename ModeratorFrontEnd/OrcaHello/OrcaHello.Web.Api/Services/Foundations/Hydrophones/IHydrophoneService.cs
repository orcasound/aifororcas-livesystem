namespace OrcaHello.Web.Api.Services
{
    public interface IHydrophoneService
    {
        ValueTask<QueryableHydrophoneData> RetrieveAllHydrophonesAsync();
    }
}

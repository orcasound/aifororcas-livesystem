namespace OrcaHello.Web.Api.Services
{
    public interface IHydrophoneOrchestrationService
    {
        ValueTask<HydrophoneListResponse> RetrieveHydrophoneLocations();
    }
}

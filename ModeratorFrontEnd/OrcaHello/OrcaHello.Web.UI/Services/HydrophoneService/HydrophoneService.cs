namespace OrcaHello.Web.UI.Services
{
    // HydrophoneService is a partial class that implements the contract for a Foundation Service that accesses an
    // API broker for hydrophone data. A Foundation Service provides basic functionality for other services, such as
    // logging or authentication. The HydrophoneService constructor uses dependency injection to receive an
    // IDetectionAPIBroker and an ILogger<HydrophoneService>. 
    public partial class HydrophoneService : IHydrophoneService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<HydrophoneService> _logger;

        public HydrophoneService(
            IDetectionAPIBroker apiBroker,
            ILogger<HydrophoneService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        // Retrieve all hydrophones asynchronously from the API broker using a TryCatch block.
        public ValueTask<List<Hydrophone>> RetrieveAllHydrophonesAsync() =>
        TryCatch(async () => {

            HydrophoneListResponse response = await _apiBroker.GetAllHydrophonesAsync();
            ValidateResponse(response);
            ValidateThereAreHydrophones(response.Count);

            return response.Hydrophones;
        });
    }
}

namespace OrcaHello.Web.UI.Services
{
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

        public ValueTask<List<Hydrophone>> RetrieveAllHydrophonesAsync() =>
        TryCatch(async () =>
        {
            HydrophoneListResponse response = await _apiBroker.GetAllHydrophonesAsync();
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response.Hydrophones;
        });
    }
}

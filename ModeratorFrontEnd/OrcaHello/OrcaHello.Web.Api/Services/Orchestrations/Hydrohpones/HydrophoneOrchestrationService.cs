namespace OrcaHello.Web.Api.Services
{
    public partial class HydrophoneOrchestrationService : IHydrophoneOrchestrationService
    {
        private readonly AppSettings _appSettings;
        private readonly ILogger<HydrophoneOrchestrationService> _logger;

        // Needed for unit testing wrapper to work properly

        public HydrophoneOrchestrationService() { }

        public HydrophoneOrchestrationService(AppSettings appSettings,
            ILogger<HydrophoneOrchestrationService> logger)
        {
            _appSettings = appSettings;
            _logger = logger;
        }

        public HydrophoneListResponse RetrieveHydrophoneLocations() =>
        TryCatch(() =>
        {
            return new HydrophoneListResponse
            {
                Hydrophones = _appSettings.HydrophoneLocations.
                    Select(h => AsHydrophone(h)).OrderBy(h => h.Name).ToList(),
                Count = _appSettings.HydrophoneLocations.Count()
            };
        });

        private Hydrophone AsHydrophone(HydrophoneLocation hydrophoneLocation)
        {
            return new Hydrophone
            {
                Id = hydrophoneLocation.Id,
                Name = hydrophoneLocation.Name,
                Longitude = hydrophoneLocation.Longitude,
                Latitude = hydrophoneLocation.Latitude
            };
        }
    }
}

namespace OrcaHello.Web.UI.Services
{
    public partial class MetricsService : IMetricsService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<MetricsService> _logger;

        public MetricsService(
            IDetectionAPIBroker apiBroker,
            ILogger<MetricsService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        public ValueTask<MetricsResponse> RetrieveFilteredMetricsAsync(DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () =>
        {
            // ValidateRequiredProperties();

            var queryString = $"fromDate={fromDate.Value.ToString()}&toDate={toDate.Value.ToString()}";

            MetricsResponse response = await _apiBroker.GetFilteredMetricsAsync(queryString);
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response;
        });
    }
}

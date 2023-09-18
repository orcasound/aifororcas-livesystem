namespace OrcaHello.Web.Api.Services
{
    public partial class MetricsOrchestrationService : IMetricsOrchestrationService
    {
        private readonly IMetadataService _metadataService;
        private readonly ILogger<MetricsOrchestrationService> _logger;

        // Needed for unit testing wrapper to work properly

        public MetricsOrchestrationService() { }

        public MetricsOrchestrationService(IMetadataService metadataService,
            ILogger<MetricsOrchestrationService> logger)
        {
            _metadataService = metadataService;
            _logger = logger;
        }

        public ValueTask<MetricsResponse> RetrieveMetricsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () =>
        {
            Validate(fromDate, nameof(fromDate));
            Validate(toDate, nameof(toDate));

            DateTime nonNullableFromDate = fromDate ?? default;
            DateTime nonNullableToDate = toDate ?? default;

            MetricsSummaryForTimeframe results = await _metadataService.
                RetrieveMetricsForGivenTimeframeAsync(nonNullableFromDate, nonNullableToDate);

            return new MetricsResponse
            {
                Unreviewed = results.QueryableRecords.Where(x => x.State == "Unreviewed").Select(x => x.Count).FirstOrDefault(),
                Positive = results.QueryableRecords.Where(x => x.State == "Positive").Select(x => x.Count).FirstOrDefault(),
                Negative = results.QueryableRecords.Where(x => x.State == "Negative").Select(x => x.Count).FirstOrDefault(),
                Unknown = results.QueryableRecords.Where(x => x.State == "Unknown").Select(x => x.Count).FirstOrDefault(),
                FromDate = results.FromDate,
                ToDate = results.ToDate
            };
        });

    }
}

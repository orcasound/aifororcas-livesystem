namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MetricsService"/> foundation service class for interacting 
    /// with metrics-related endpoints of the API.
    /// </summary>
    /// <param name="apiBroker">The detection API broker.</param>
    /// <param name="logger">The logger.</param>
    public partial class MetricsService : IMetricsService
    {
        private readonly IDetectionAPIBroker _apiBroker = null!;
        private readonly ILogger<MetricsService> _logger = null!;


        // Needed for unit testing wrapper to work properly

        public MetricsService() { }

        public MetricsService(
            IDetectionAPIBroker apiBroker,
            ILogger<MetricsService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        /// <summary>
        /// Retrieves the metrics counts (unreviewed, positive, negative, unknown), filtered by the date range 
        /// from the API.
        /// </summary>
        /// <param name="fromDate">The start date of the filter.</param>
        /// <param name="toDate">The end date of the filter.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="InvalidMetricsException">If the date range is invalid.</exception>
        /// <exception cref="NullMetricsResponseException">If the response from the API is null.</exception>
        public ValueTask<MetricsResponse> RetrieveFilteredMetricsAsync(DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () =>
        {
            ValidateDateRange(fromDate, toDate);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";

            MetricsResponse response = await _apiBroker.GetFilteredMetricsAsync(queryString);

            ValidateResponse(response);

            return response;
        });
    }
}

﻿namespace OrcaHello.Web.UI.Services
{
    public partial class DetectionService : IDetectionService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<DetectionService> _logger;

        public DetectionService(
            IDetectionAPIBroker apiBroker,
            ILogger<DetectionService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        public ValueTask<DetectionListResponse> RetrieveFilteredAndPaginatedDetectionsAsync(string state, DateTime? fromDate, DateTime? toDate,
            string sortBy, bool isDescending, int page, int pageSize, string location) =>
        TryCatch(async () =>
        {
            // ValidateRequiredProperties();

            var queryString = $"state={state}&fromDate={fromDate.Value.ToString()}&toDate={toDate.Value.ToString()}&sortBy={sortBy}&isDescending={isDescending}";
            queryString += $"&page={page}&pageSize={pageSize}";
            if (!string.IsNullOrWhiteSpace(location))
                queryString += $"&location={location}";

            DetectionListResponse response = await _apiBroker.GetFilteredDetectionsAsync(queryString);
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response;
        });
    }
}
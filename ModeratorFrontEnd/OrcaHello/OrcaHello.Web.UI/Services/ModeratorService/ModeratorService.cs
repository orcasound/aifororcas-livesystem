namespace OrcaHello.Web.UI.Services
{
    // ModeratorService is a partial class that implements the contract for a Foundation Service that accesses an
    // API broker for moderator-related data. A Foundation Service provides basic functionality for other services, such as
    // logging or authentication. The ModeratorService constructor uses dependency injection to receive an
    // IDetectionAPIBroker and an ILogger<ModeratorService>. 
    public partial class ModeratorService : IModeratorService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<ModeratorService> _logger;

        public ModeratorService(
        IDetectionAPIBroker apiBroker,
        ILogger<ModeratorService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }
        // Returns a list of positive comments for a given moderator, filtered by the date range, page number, and page size.
        // It calls the broker service, validates the parameters and the response, and handles any exceptions.
        public ValueTask<CommentListForModeratorResponse> GetFilteredPositiveCommentsForModeratorAsync(string moderator,
        DateTime? fromDate, DateTime? toDate, int page, int pageSize) =>
        TryCatch(async () => {
            ValidateModerator(moderator);
            ValidateDateRange(fromDate, toDate);
            ValidatePagination(page, pageSize);
            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";
            queryString += $"&page={page}&pageSize={pageSize}";
            CommentListForModeratorResponse response = await _apiBroker
                .GetFilteredPositiveCommentsForModeratorAsync(moderator, queryString);

            ValidateCommentResponse(response);

            return response;
        });

        // Returns a list of negative and unknown comments for a given moderator, filtered by the date range, page number, and page size.
        // It calls the broker service, validates the parameters and the response, and handles any exceptions.
        public ValueTask<CommentListForModeratorResponse> GetFilteredNegativeAndUknownCommentsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate, int page, int pageSize) =>
        TryCatch(async () => {

            ValidateModerator(moderator);
            ValidateDateRange(fromDate, toDate);
            ValidatePagination(page, pageSize);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";
            queryString += $"&page={page}&pageSize={pageSize}";

            CommentListForModeratorResponse response = await _apiBroker
                .GetFilteredNegativeAndUknownCommentsForModeratorAsync(moderator, queryString);

            ValidateCommentResponse(response);
            return response;
        });

        // Returns a list of tags for a given moderator, filtered by the date range.
        // It calls the broker service, validates the parameters and the response, and handles any exceptions.
        public ValueTask<TagListForModeratorResponse> GetFilteredTagsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () => {

            ValidateModerator(moderator);
            ValidateDateRange(fromDate, toDate);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";

             TagListForModeratorResponse response = await _apiBroker
                .GetFilteredTagsForModeratorAsync(moderator, queryString);

            ValidateTagResponse(response);

            return response;
        });

        // Returns a list of detections for a given moderator and tag, filtered by the date range, page number, and page size.
        // It calls the broker service, validates the parameters and the response, and handles any exceptions.
        public ValueTask<DetectionListForModeratorAndTagResponse> GetFilteredDetectionsForTagAndModeratorAsync(string moderator, string tag,
            DateTime? fromDate, DateTime? toDate, int page, int pageSize) =>
        TryCatch(async () => {
            ValidateTag(tag);
            ValidateModerator(moderator);
            ValidateDateRange(fromDate, toDate);
            ValidatePagination(page, pageSize);
            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";
            queryString += $"&page={page}&pageSize={pageSize}";
            DetectionListForModeratorAndTagResponse response = await _apiBroker
                .GetFilteredDetectionsForTagAndModeratorAsync(moderator, tag, queryString);

            ValidateDetectionResponse(response);

            return response;
        });

        // Returns a metrics object for a given moderator, filtered by the date range.
        // It calls the broker service, validates the parameters and the response, and handles any exceptions.
        public ValueTask<MetricsForModeratorResponse> GetFilteredMetricsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () => {

            ValidateModerator(moderator);
            ValidateDateRange(fromDate, toDate);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";

            MetricsForModeratorResponse response = await _apiBroker.GetFilteredMetricsForModeratorAsync(moderator, queryString);

            ValidateMetricsResponse(response);

            return response;
        });
    }
}

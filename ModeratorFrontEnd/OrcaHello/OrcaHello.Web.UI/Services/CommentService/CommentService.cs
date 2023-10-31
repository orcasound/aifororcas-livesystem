using Azure;

namespace OrcaHello.Web.UI.Services
{
    public partial class CommentService : ICommentService
    {
        private readonly IDetectionAPIBroker _apiBroker;
        private readonly ILogger<CommentService> _logger;

        public CommentService(
            IDetectionAPIBroker apiBroker,
            ILogger<CommentService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        public ValueTask<CommentListResponse> RetrieveFilteredPositiveCommentsAsync(DateTime? fromDate, DateTime? toDate,
            int page, int pageSize) =>
        TryCatch(async () =>
        {
            // ValidateRequiredProperties();

            var queryString = $"fromDate={fromDate.Value.ToString()}&toDate={toDate.Value.ToString()}";
            queryString += $"&page={page}&pageSize={pageSize}";

            CommentListResponse response = await _apiBroker.GetFilteredPositiveCommentsAsync(queryString);
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response;
        });

        public ValueTask<CommentListResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(DateTime? fromDate, DateTime? toDate,
            int page, int pageSize) =>
        TryCatch(async () =>
        {
            // ValidateRequiredProperties();

            var queryString = $"fromDate={fromDate.Value.ToString()}&toDate={toDate.Value.ToString()}";
            queryString += $"&page={page}&pageSize={pageSize}";

            CommentListResponse response = await _apiBroker.GetFilteredNegativeAndUknownCommentsAsync(queryString);
            //ValidateResponseNotNull();
            //ValidateResponseHasValues();

            return response;
        });
    }
}

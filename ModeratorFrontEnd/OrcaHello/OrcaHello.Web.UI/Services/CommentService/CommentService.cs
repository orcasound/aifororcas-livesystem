namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CommentService"/> foundation service class for interacting 
    /// with comment-related endpoints of the API.
    /// </summary>
    /// <param name="apiBroker">The detection API broker.</param>
    /// <param name="logger">The logger.</param>
    public partial class CommentService : ICommentService
    {
        private readonly IDetectionAPIBroker _apiBroker = null!;
        private readonly ILogger<CommentService> _logger = null!;

        // Needed for unit testing wrapper to work properly

        public CommentService() { }

        public CommentService(
            IDetectionAPIBroker apiBroker,
            ILogger<CommentService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        /// <summary>
        /// Retrieves a list of positive comments, filtered by the date range, page number, 
        /// and page size from the API.
        /// </summary>
        /// <param name="fromDate">The start date of the filter.</param>
        /// <param name="toDate">The end date of the filter.</param>
        /// <param name="page">Which page of the list to return.</param>
        /// <param name="pageSize">The page size to return.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="InvalidCommentException">If one or more of the parameters are invalid.</exception>
        /// <exception cref="NullCommentResponseException">If the response from the API is null.</exception>
        public ValueTask<CommentListResponse> RetrieveFilteredPositiveCommentsAsync(DateTime? fromDate, DateTime? toDate,
            int page, int pageSize) =>
        TryCatch(async () =>
        {
            ValidateDateRange(fromDate, toDate);
            ValidatePagination(page, pageSize);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";
            queryString += $"&page={page}&pageSize={pageSize}";

            CommentListResponse response = await _apiBroker.GetFilteredPositiveCommentsAsync(queryString);

            ValidateResponse(response);

            return response;
        });

        /// <summary>
        /// Retrieves a list of negative and unknown, filtered by the date range, page number, 
        /// and page size from the API.
        /// </summary>
        /// <param name="fromDate">The start date of the filter.</param>
        /// <param name="toDate">The end date of the filter.</param>
        /// <param name="page">Which page of the list to return.</param>
        /// <param name="pageSize">The page size to return.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="InvalidCommentException">If one or more of the parameters are invalid.</exception>
        /// <exception cref="NullCommentResponseException">If the response from the API is null.</exception>
        public ValueTask<CommentListResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(DateTime? fromDate, DateTime? toDate,
            int page, int pageSize) =>
        TryCatch(async () =>
        {
            ValidateDateRange(fromDate, toDate);
            ValidatePagination(page, pageSize);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";
            queryString += $"&page={page}&pageSize={pageSize}";

            CommentListResponse response = await _apiBroker.GetFilteredNegativeAndUknownCommentsAsync(queryString);

            ValidateResponse(response);

            return response;
        });
    }
}

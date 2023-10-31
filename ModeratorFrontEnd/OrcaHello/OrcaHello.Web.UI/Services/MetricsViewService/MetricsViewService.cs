namespace OrcaHello.Web.UI.Services
{
    public partial class MetricsViewService : IMetricsViewService
    {
        private readonly IDetectionService _detectionService;
        private readonly ITagService _tagService;
        private readonly IMetricsService _metricsService;
        private readonly ICommentService _commentService;
        private readonly ILogger<MetricsViewService> _logger;

        public MetricsViewService(IDetectionService detectionService,
            ITagService tagService,
            IMetricsService metricsService,
            ICommentService commentService,
            ILogger<MetricsViewService> logger)
        {
            _detectionService = detectionService;
            _tagService = tagService;
            _metricsService = metricsService;
            _commentService = commentService;
            _logger = logger;
        }

        public ValueTask<List<string>> RetrieveFilteredTagsAsync(TagsByDateRequest request) =>
        TryCatch(async () =>
        {
            TagListForTimeframeResponse response =
                await _tagService.RetrieveFilteredTagsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate);

            return response.Tags;
        });

        public ValueTask<MetricsItemViewResponse> RetrieveFilteredMetricsAsync(MetricsByDateRequest request) =>
        TryCatch(async () =>
        {
            MetricsResponse response =
                await _metricsService.RetrieveFilteredMetricsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate);

            return MetricsItemView.AsMetricsItemViewResponse(response);
        });

        public ValueTask<DetectionItemViewResponse> RetrieveFilteredDetectionsForTagsAsync(PaginatedDetectionsByTagAndDateRequest request) =>
        TryCatch(async () =>
        {
            DetectionListForTagResponse response =
                await _detectionService.RetrieveFilteredAndPaginatedDetectionsForTagAsync(
                    tag: request.Tag,
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            return new DetectionItemViewResponse
            {
                DetectionItemViews = response.Detections.Select(DetectionItemView.AsDetectionItemView).ToList(),
                Count = response.TotalCount
            };
        });

        public ValueTask<CommentItemViewResponse> RetrieveFilteredPositiveCommentsAsync(PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            CommentListResponse response =
                await _commentService.RetrieveFilteredPositiveCommentsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            return new CommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount
            };
        });

        public ValueTask<CommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            CommentListResponse response =
                await _commentService.RetrieveFilteredNegativeAndUnknownCommentsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            return new CommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount
            };
        });
    }
}

namespace OrcaHello.Web.UI.Services
{
    public partial class DetectionViewService : IDetectionViewService
    {
        private readonly IDetectionService _detectionService;
        private readonly ILogger<DetectionViewService> _logger;

        public DetectionViewService(IDetectionService detectionService,
            ILogger<DetectionViewService> logger)
        {
            _detectionService = detectionService;
            _logger = logger;
        }

        public ValueTask<DetectionItemViewResponse> RetrieveFilteredAndPaginatedDetectionItemViewsAsync(DetectionFilterAndPagination options) =>
        TryCatch(async () =>
        {
            DetectionListResponse detectionListResponse =
                await _detectionService.RetrieveFilteredAndPaginatedDetectionsAsync(
                    state: options.State,
                    fromDate: options.FromDate,
                    toDate: options.ToDate,
                    sortBy: options.SortBy,
                    isDescending: options.IsDescending,
                    location: options.Location,
                    page: options.Page,
                    pageSize: options.PageSize);

            return new DetectionItemViewResponse
            {
                DetectionItemViews = detectionListResponse.Detections.Select(AsDetectionItemView).ToList(),
                Count = detectionListResponse.TotalCount
            };
        });

        private static Func<Detection, DetectionItemView> AsDetectionItemView =>
            detection => new DetectionItemView
            {
                Id = detection.Id,
                LocationName = detection.LocationName,
                Timestamp = detection.Timestamp,
                AudioUri = detection.AudioUri,
                SpectrogramUri = detection.SpectrogramUri,
                Confidence = detection.Confidence,
                State = detection.State,
                Location = new() { 
                    Name = detection.Location.Name,
                    Longitude = detection.Location.Longitude,
                    Latitude = detection.Location.Latitude },
                Comments = detection.Comments,
                Moderator = detection.Moderator,
                Moderated = detection.Moderated,
                Tags = String.Join(", ", detection.Tags),
                Annotations = detection.Annotations.Select(AsAnnotationItemView).ToList(),
                InterestLabel = detection.InterestLabel
            };

        private static Func<Annotation, AnnotationItemView> AsAnnotationItemView =>
            annotation => new AnnotationItemView
            {
                Id = annotation.Id,
                StartTime = annotation.StartTime,
                EndTime = annotation.EndTime,
                Confidence = annotation.Confidence
            };
    }

}


namespace OrcaHello.Web.UI.Services
{
    public partial class MetricsViewService : IMetricsViewService
    {
        private readonly IDetectionService _detectionService;
        private readonly ITagService _tagService;
        private readonly ILogger<MetricsViewService> _logger;

        public MetricsViewService(IDetectionService detectionService,
            ITagService tagService,
            ILogger<MetricsViewService> logger)
        {
            _detectionService = detectionService;
            _tagService = tagService;
            _logger = logger;
        }

        public ValueTask<List<string>> RetrieveFilteredTagsAsync(TagFilter options) =>
        TryCatch(async () =>
        {
            TagListForTimeframeResponse tagListForTimeframeResponse =
                await _tagService.RetrieveFilteredTagsAsync(
                    fromDate: options.FromDate,
                    toDate: options.ToDate);

            return tagListForTimeframeResponse.Tags;
        });
    }
}

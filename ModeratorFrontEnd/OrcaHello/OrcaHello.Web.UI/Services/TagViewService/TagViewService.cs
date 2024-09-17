namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TagViewService"/> orchestration service class for rendering 
    /// data from TagService and DetectionService for various tag-related views and implementing tag-related business logic.
    /// </summary>
    /// <param name="tagService">The TagService foundation service.</param>
    /// <param name="detectionService"> The DetectionService foundation service.</param>
    /// <param name="logger">The logger.</param>
    public partial class TagViewService : ITagViewService
    {
        private readonly ITagService _tagService = null!;
        private readonly IDetectionService _detectionService = null!;

        private readonly ILogger<TagViewService> _logger = null!;

        // Needed for unit testing wrapper to work properly

        public TagViewService() { }

        public TagViewService(
            ITagService tagService,
            IDetectionService detectionService,
            ILogger<TagViewService> logger)
        {
            _tagService = tagService;
            _detectionService = detectionService;
            _logger = logger;
        }

        /// <summary>
        /// Retrieves a list of all tags from TagService and sorts them alphabetically.
        /// </summary>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagViewResponseException">If the response from TagService is null.</exception>
        public ValueTask<List<TagItemView>> RetrieveAllTagViewsAsync() =>
        TryCatch(async () =>
        {
            List<string> response =
                await _tagService.RetrieveAllTagsAsync();

            ValidateResponse(response);

            return response.Select(AsTagView).OrderBy(x => x.Tag).ToList();
        });

        /// <summary>
        /// Initiates the delete of all instances of a tag in all detections in the database.
        /// </summary>
        /// <param name="tagItem">The tag being deleted.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagViewException">If the TagItem is null.</exception>
        /// <exception cref="InvalidTagViewException">If the Tag property is null.</exception>
        /// <exception cref="NullTagViewResponseException">If the response from TagService is null.</exception>
        public ValueTask<TagItemViewResponse> DeleteTagAsync(TagItemView tagItem) =>
        TryCatch(async () =>
        {
            ValidateRequest(tagItem);
            ValidateTagString(tagItem.Tag, "'Tag'");

            TagRemovalResponse response =
                await _tagService.RemoveTagAsync(tagItem.Tag);

            ValidateResponse(response);

            return new TagItemViewResponse
            {
                MatchingTags = response.TotalMatching,
                ProcessedTags = response.TotalRemoved,
            };
        });

        /// <summary>
        /// Initiates the replace of all instances of the old tag in all detections in the database with the new tag.
        /// </summary>
        /// <param name="request">A request containing the old and new tags.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagViewRequestException">If the request is null.</exception>
        /// <exception cref="InvalidTagViewException">If either of the tags are null or empty.</exception>
        /// <exception cref="NullTagViewResponseException">If the response from TagService is null.</exception>
        public ValueTask<TagItemViewResponse> ReplaceTagAsync(ReplaceTagRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateTagString(request.OldTag, "Old 'Tag'");
            ValidateTagString(request.NewTag, "New 'Tag'");

            TagReplaceResponse response =
                await _tagService.ReplaceTagAsync(request);

            ValidateResponse(response);

            return new TagItemViewResponse
            {
                MatchingTags = response.TotalMatching,
                ProcessedTags = response.TotalReplaced
            };
        });

        /// <summary>
        /// Retrieves a list of detections from DetectionService for a specific tag string, filtered by the date range
        /// and paginated by the page number and size.
        /// </summary>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagViewRequestException">If the request is null.</exception>
        /// <exception cref="InvalidTagViewException">If the request is improperly formatted.</exception>
        /// <exception cref="NullTagViewResponseException">If the response from DetectionService is null.</exception>
        public ValueTask<DetectionItemViewResponse> RetrieveDetectionsByTagsAsync(PaginatedDetectionsByTagsAndDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateAtLeastOneTagSelected(request.Tags);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            var tagString = request.Logic == LogicalOperator.Or ?
                string.Join("|", request.Tags) :
                string.Join(",", request.Tags);

            DetectionListForTagResponse response =
            await _detectionService.RetrieveFilteredAndPaginatedDetectionsForTagAsync(
            tag: tagString,
            fromDate: request.FromDate,
            toDate: request.ToDate,
            page: request.Page,
            pageSize: request.PageSize);

            ValidateResponse(response);

            return new DetectionItemViewResponse
            {
                DetectionItemViews = response.Detections.Select(DetectionItemView.AsDetectionItemView).ToList(),
                Count = response.TotalCount
            };
        });

        // Maps a tag string to a tag view for presentation.
        private static Func<string, TagItemView> AsTagView =>
            tag => new TagItemView
            {
                Tag = tag
            };
    }
}

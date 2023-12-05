namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TagService"/> foundation service class for interacting 
    /// with tag-related endpoints of the API.
    /// </summary>
    /// <param name="apiBroker">The detection API broker.</param>
    /// <param name="logger">The logger.</param>
    public partial class TagService : ITagService
    {
        private readonly IDetectionAPIBroker _apiBroker = null!;
        private readonly ILogger<TagService> _logger = null!;

        // Needed for unit testing wrapper to work properly

        public TagService() { }

        public TagService(
            IDetectionAPIBroker apiBroker,
            ILogger<TagService> logger)
        {
            _apiBroker = apiBroker;
            _logger = logger;
        }

        /// <summary>
        /// Retrieves a list of all tags.
        /// </summary>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagResponseException">If the response from the API is null.</exception>
        public ValueTask<List<string>> RetrieveAllTagsAsync() =>
        TryCatch(async () =>
        {
            TagListResponse response = await _apiBroker.GetAllTagsAsync();

            ValidateResponse(response);

            return response.Tags;
        });

        /// <summary>
        /// Retrieves a list of tags, filtered by the date range from the API.
        /// </summary>
        /// <param name="fromDate">The start date of the filter.</param>
        /// <param name="toDate">The end date of the filter.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="InvalidTagException">If one or more of the parameters are invalid.</exception>
        /// <exception cref="NullTagResponseException">If the response from the API is null.</exception>
        public ValueTask<TagListForTimeframeResponse> RetrieveFilteredTagsAsync(DateTime? fromDate, DateTime? toDate) =>
        TryCatch(async () =>
        {
            ValidateDateRange(fromDate, toDate);

            var queryString = $"fromDate={fromDate.GetValueOrDefault()}&toDate={toDate.GetValueOrDefault()}";

            TagListForTimeframeResponse response = await _apiBroker.GetFilteredTagsAsync(queryString);

            ValidateResponse(response);

            return response;
        });

        /// <summary>
        /// Calls the API to remove all instances of a tag.
        /// </summary>
        /// <param name="tag">The tag to remove.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="InvalidTagException">If the tag parameter is invalid.</exception>
        /// <exception cref="NullTagResponseException">If the response from the API is null.</exception>
        public ValueTask<TagRemovalResponse> RemoveTagAsync(string tag) =>
        TryCatch(async () =>
        {
            Validate(tag, nameof(tag));

            TagRemovalResponse response = await _apiBroker.RemoveTag(tag);

            ValidateResponse(response);

            return response;
        });

        /// <summary>
        /// Calls the API to replace all instances of one tag with another.
        /// </summary>
        /// <param name="request">Request containing the old and new tag.</param>
        /// <returns>A <see cref="ValueTask{TResult}"/> that represents the asynchronous operation.</returns>
        /// <exception cref="NullTagRequestException">If request is null.</exception>
        /// <exception cref="InvalidTagException">If one or more of the parameters are invalid.</exception>
        /// <exception cref="NullTagResponseException">If the response from the API is null.</exception>
        public ValueTask<TagReplaceResponse> ReplaceTagAsync(ReplaceTagRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            Validate(request.OldTag, nameof(request.OldTag));
            Validate(request.NewTag, nameof(request.NewTag));

            TagReplaceResponse response = await _apiBroker.ReplaceTagAsync(request);

            ValidateResponse(response);

            return response;
        });
    }
}

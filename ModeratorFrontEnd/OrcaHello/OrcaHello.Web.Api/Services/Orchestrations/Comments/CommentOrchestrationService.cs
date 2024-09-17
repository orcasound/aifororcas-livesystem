namespace OrcaHello.Web.Api.Services
{
    public partial class CommentOrchestrationService : ICommentOrchestrationService
    {
        private readonly IMetadataService _metadataService;
        private readonly ILogger<CommentOrchestrationService> _logger;

        // Needed for unit testing wrapper to work properly

        public CommentOrchestrationService() { }

        public CommentOrchestrationService(IMetadataService metadataService,
            ILogger<CommentOrchestrationService> logger)
        {
            _metadataService = metadataService;
            _logger = logger;
        }

        public ValueTask<CommentListResponse> RetrievePositiveCommentsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate, int page, int pageSize) =>
        TryCatch(async () =>
        {
            Validate(fromDate, nameof(fromDate));
            Validate(toDate, nameof(toDate));
            ValidatePage(page);
            ValidatePageSize(pageSize);

            DateTime nonNullableFromDate = fromDate ?? default;
            DateTime nonNullableToDate = toDate ?? default;

            QueryableMetadataForTimeframe results = await _metadataService.
                RetrievePositiveMetadataForGivenTimeframeAsync(nonNullableFromDate, nonNullableToDate, page, pageSize);

            return new CommentListResponse
            {
                Comments = results.QueryableRecords.Select(r => AsComment(r)).ToList(),
                FromDate = results.FromDate,
                ToDate = results.ToDate,
                Page = results.Page,
                PageSize = results.PageSize,
                TotalCount = results.TotalCount,
                Count = results.QueryableRecords.Count()
            };
        });

        public ValueTask<CommentListResponse> RetrieveNegativeAndUnknownCommentsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate, int page, int pageSize) =>
         TryCatch(async () =>
         {
             Validate(fromDate, nameof(fromDate));
             Validate(toDate, nameof(toDate));
             ValidatePage(page);
             ValidatePageSize(pageSize);

             DateTime nonNullableFromDate = fromDate ?? default;
             DateTime nonNullableToDate = toDate ?? default;

             QueryableMetadataForTimeframe results = await _metadataService.
                 RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync(nonNullableFromDate, nonNullableToDate, page, pageSize);

             return new CommentListResponse
             {
                 Comments = results.QueryableRecords.Select(r => AsComment(r)).ToList(),
                 FromDate = results.FromDate,
                 ToDate = results.ToDate,
                 Page = results.Page,
                 PageSize = results.PageSize,
                 TotalCount = results.TotalCount,
                 Count = results.QueryableRecords.Count()
             };
         });

        private Comment AsComment(Metadata metadata)
        {
            var comment = new Comment
            {
                Id = metadata.Id,
                Comments = metadata.Comments,
                LocationName = metadata.LocationName,
                Moderator = metadata.Moderator,
                Moderated = !string.IsNullOrWhiteSpace(metadata.DateModerated) ? DateTime.Parse(metadata.DateModerated) : null,
                Timestamp = metadata.Timestamp,
                SpectrogramUri = metadata.ImageUri,
                AudioUri = metadata.AudioUri
            };

            return comment;
        }
    }
}
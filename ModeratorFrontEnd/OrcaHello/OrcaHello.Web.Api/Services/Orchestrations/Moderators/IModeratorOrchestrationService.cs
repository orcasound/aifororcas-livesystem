namespace OrcaHello.Web.Api.Services
{
    public interface IModeratorOrchestrationService
    {
        ValueTask<ModeratorListResponse> RetrieveModeratorsAsync();
        ValueTask<MetricsForModeratorResponse> RetrieveMetricsForGivenTimeframeAndModeratorAsync(DateTime? fromDate, DateTime? toDate, string moderator);
        ValueTask<TagListForModeratorResponse> RetrieveTagsForGivenTimePeriodAndModeratorAsync(DateTime? fromDate, DateTime? toDate, string moderator);
        ValueTask<CommentListForModeratorResponse> RetrievePositiveCommentsForGivenTimeframeAndModeratorAsync(DateTime? fromDate, DateTime? toDate, string moderator, int page, int pageSize);
        ValueTask<CommentListForModeratorResponse> RetrieveNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync(DateTime? fromDate, DateTime? toDate, string moderator, int page, int pageSize);
        ValueTask<DetectionListForModeratorAndTagResponse> RetrieveDetectionsForGivenTimeframeTagAndModeratorAsync(DateTime? fromDate, DateTime? toDate, string moderator, string tag, int page, int pageSize);
    }
}

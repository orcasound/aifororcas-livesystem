namespace OrcaHello.Web.UI.Services
{
    public interface IModeratorService
    {
        ValueTask<CommentListForModeratorResponse> GetFilteredPositiveCommentsForModeratorAsync(string moderator,
            DateTime? fromDate, DateTime? toDate, int page, int pageSize);

        ValueTask<CommentListForModeratorResponse> GetFilteredNegativeAndUknownCommentsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate, int page, int pageSize);

        ValueTask<TagListForModeratorResponse> GetFilteredTagsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate);

        ValueTask<MetricsForModeratorResponse> GetFilteredMetricsForModeratorAsync(string moderator,
           DateTime? fromDate, DateTime? toDate);

        ValueTask<DetectionListForModeratorAndTagResponse> GetFilteredDetectionsForTagAndModeratorAsync(string moderator, string tag,
            DateTime? fromDate, DateTime? toDate, int page, int pageSize);
    }
}

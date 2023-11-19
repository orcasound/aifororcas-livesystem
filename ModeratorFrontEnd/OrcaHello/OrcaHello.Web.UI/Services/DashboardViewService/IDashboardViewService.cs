namespace OrcaHello.Web.UI.Services
{
    public interface IDashboardViewService
    {
        ValueTask<List<string>> RetrieveFilteredTagsAsync(TagsByDateRequest request);
        ValueTask<MetricsItemViewResponse> RetrieveFilteredMetricsAsync(MetricsByDateRequest request);
        ValueTask<DetectionItemViewResponse> RetrieveFilteredDetectionsForTagsAsync(PaginatedDetectionsByTagAndDateRequest request);
        ValueTask<CommentItemViewResponse> RetrieveFilteredPositiveCommentsAsync(PaginatedCommentsByDateRequest request);
        ValueTask<CommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(PaginatedCommentsByDateRequest request);
        ValueTask<List<string>> RetrieveFilteredTagsForModeratorAsync(string moderator, TagsByDateRequest request);
        ValueTask<ModeratorMetricsItemViewResponse> RetrieveFilteredMetricsForModeratorAsync(string moderator, MetricsByDateRequest request);
        ValueTask<ModeratorCommentItemViewResponse> RetrieveFilteredPositiveCommentsForModeratorAsync(string moderator, PaginatedCommentsByDateRequest request);
        ValueTask<ModeratorCommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsForModeratorAsync(string moderator, PaginatedCommentsByDateRequest request);
        ValueTask<ModeratorDetectionItemViewResponse> RetrieveFilteredDetectionsForTagAndModeratorAsync(string moderator, PaginatedDetectionsByTagAndDateRequest request);
    }
}

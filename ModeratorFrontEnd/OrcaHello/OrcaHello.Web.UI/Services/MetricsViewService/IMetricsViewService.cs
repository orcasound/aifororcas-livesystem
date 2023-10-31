namespace OrcaHello.Web.UI.Services
{
    public interface IMetricsViewService
    {
        ValueTask<List<string>> RetrieveFilteredTagsAsync(TagsByDateRequest request);
        ValueTask<MetricsItemViewResponse> RetrieveFilteredMetricsAsync(MetricsByDateRequest request);
        ValueTask<DetectionItemViewResponse> RetrieveFilteredDetectionsForTagsAsync(PaginatedDetectionsByTagAndDateRequest request);
        ValueTask<CommentItemViewResponse> RetrieveFilteredPositiveCommentsAsync(PaginatedCommentsByDateRequest request);
        ValueTask<CommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(PaginatedCommentsByDateRequest request);
    }
}

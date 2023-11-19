namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<CommentListForModeratorResponse> GetFilteredPositiveCommentsForModeratorAsync(string moderator, string queryString);
        ValueTask<CommentListForModeratorResponse> GetFilteredNegativeAndUknownCommentsForModeratorAsync(string moderator, string queryString);
        ValueTask<TagListForModeratorResponse> GetFilteredTagsForModeratorAsync(string moderator, string queryString);
        ValueTask<MetricsForModeratorResponse> GetFilteredMetricsForModeratorAsync(string moderator, string queryString);
        ValueTask<DetectionListForModeratorAndTagResponse> GetFilteredDetectionsForTagAndModeratorAsync(string moderator, string tag, string queryString);
    }
}

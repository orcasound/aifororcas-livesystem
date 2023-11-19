namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string moderatorsRelativeUrl = "moderators";

        public async ValueTask<CommentListForModeratorResponse> GetFilteredPositiveCommentsForModeratorAsync(string moderator, string queryString) =>
            await this.GetAsync<CommentListForModeratorResponse>($"{moderatorsRelativeUrl}/{moderator}/comments/positive?{queryString}");

        public async ValueTask<CommentListForModeratorResponse> GetFilteredNegativeAndUknownCommentsForModeratorAsync(string moderator, string queryString) =>
            await this.GetAsync<CommentListForModeratorResponse>($"{moderatorsRelativeUrl}/{moderator}/comments/negative-unknown?{queryString}");

        public async ValueTask<TagListForModeratorResponse> GetFilteredTagsForModeratorAsync(string moderator, string queryString) =>
            await this.GetAsync<TagListForModeratorResponse>($"{moderatorsRelativeUrl}/{moderator}/tags?{queryString}");

        public async ValueTask<MetricsForModeratorResponse> GetFilteredMetricsForModeratorAsync(string moderator, string queryString) =>
            await this.GetAsync<MetricsForModeratorResponse>($"{moderatorsRelativeUrl}/{moderator}/metrics?{queryString}");

        public async ValueTask<DetectionListForModeratorAndTagResponse> GetFilteredDetectionsForTagAndModeratorAsync(string moderator, string tag, string queryString) =>
            await this.GetAsync<DetectionListForModeratorAndTagResponse>($"{moderatorsRelativeUrl}/{moderator}/detections/{tag}?{queryString}");
    }
}

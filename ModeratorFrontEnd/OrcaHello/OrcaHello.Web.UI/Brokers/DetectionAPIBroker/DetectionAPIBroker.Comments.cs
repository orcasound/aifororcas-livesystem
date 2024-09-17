namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string commentsRelativeUrl = "comments";

        public async ValueTask<CommentListResponse> GetFilteredPositiveCommentsAsync(string queryString) =>
            await this.GetAsync<CommentListResponse>($"{commentsRelativeUrl}/positive?{queryString}");

        public async ValueTask<CommentListResponse> GetFilteredNegativeAndUknownCommentsAsync(string queryString) =>
            await this.GetAsync<CommentListResponse>($"{commentsRelativeUrl}/negative-unknown?{queryString}");
    }
}

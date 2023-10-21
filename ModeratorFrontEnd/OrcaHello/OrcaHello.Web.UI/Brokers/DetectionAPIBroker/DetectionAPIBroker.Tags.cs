namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string tagRelativeUrl = "tags";

        public async ValueTask<TagListResponse> GetAllTagsAsync() =>
            await this.GetAsync<TagListResponse>(tagRelativeUrl);

        public async ValueTask<TagListForTimeframeResponse> GetFilteredTagsAsync(string queryString) =>
            await this.GetAsync<TagListForTimeframeResponse>($"{tagRelativeUrl}?{queryString}");
    }
}

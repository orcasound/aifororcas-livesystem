namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string tagRelativeUrl = "tags";

        public async ValueTask<TagListResponse> GetAllTagsAsync() =>
            await this.GetAsync<TagListResponse>(tagRelativeUrl);

        public async ValueTask<TagListForTimeframeResponse> GetFilteredTagsAsync(string queryString) =>
            await this.GetAsync<TagListForTimeframeResponse>($"{tagRelativeUrl}/bytimeframe?{queryString}");

        public async ValueTask<TagRemovalResponse> RemoveTag(string tag) =>
            await this.DeleteAsync<TagRemovalResponse>($"{tagRelativeUrl}/{tag}");

        public async ValueTask<TagReplaceResponse> ReplaceTagAsync(ReplaceTagRequest request) =>
            await this.PutAsync<ReplaceTagRequest, TagReplaceResponse>($"{tagRelativeUrl}/replace", request);
    }
}

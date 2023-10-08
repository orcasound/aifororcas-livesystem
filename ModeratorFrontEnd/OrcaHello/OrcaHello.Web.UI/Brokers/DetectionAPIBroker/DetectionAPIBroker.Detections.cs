namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string detectionRelativeUrl = "detections";

        public async ValueTask<DetectionListResponse> GetFilteredDetectionsAsync(string queryString) =>
            await this.GetAsync<DetectionListResponse>($"{detectionRelativeUrl}?{queryString}");

        public async ValueTask<ModerateDetectionsResponse> PutModerateDetectionsAsync(ModerateDetectionsRequest request) =>
            await this.PutAsync<ModerateDetectionsRequest, ModerateDetectionsResponse>($"{detectionRelativeUrl}/moderate", request);
    }
}

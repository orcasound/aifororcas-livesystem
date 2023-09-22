namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string detectionRelativeUrl = "detections";

        public async ValueTask<DetectionListResponse> GetFilteredDetectionsAsync(string queryString) =>
            await this.GetAsync<DetectionListResponse>($"{detectionRelativeUrl}?{queryString}");
    }
}

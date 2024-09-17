namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string metricsRelativeUrl = "metrics";

        public async ValueTask<MetricsResponse> GetFilteredMetricsAsync(string queryString) =>
            await this.GetAsync<MetricsResponse>($"{metricsRelativeUrl}?{queryString}");
    }
}

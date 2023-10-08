namespace OrcaHello.Web.UI.Brokers
{
    [ExcludeFromCodeCoverage]
    public partial class DetectionAPIBroker : IDetectionAPIBroker
    {
        // TODO: Move to an appsettings
        private readonly string _detectionApiUrn = "https://localhost:5001/api";

        private readonly IHttpService _apiClient;

        public DetectionAPIBroker(IHttpService apiClient)
        {
            _apiClient = apiClient;
        }

        private async ValueTask<T> GetAsync<T>(string relativeUrl) =>
            await _apiClient.GetContentAsync<T>(createFullUrl(relativeUrl));

        private async ValueTask<T> PostAsync<T>(string relativeUrl, T content) =>
            await _apiClient.PostContentAsync<T>(createFullUrl(relativeUrl), content);

        private async ValueTask<TResult> PutAsync<T, TResult>(string relativeUrl, T content) =>
            await _apiClient.PutContentAsync<T, TResult>(createFullUrl(relativeUrl), content);

        private string createFullUrl(string relativeUrl)
        {
            return _detectionApiUrn.EndsWith("/") ?
                $"{_detectionApiUrn}{relativeUrl}" :
                $"{_detectionApiUrn}/{relativeUrl}";
        }
    }
}

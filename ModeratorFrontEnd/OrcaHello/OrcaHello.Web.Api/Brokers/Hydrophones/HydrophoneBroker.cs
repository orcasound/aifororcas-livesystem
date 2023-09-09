namespace OrcaHello.Web.Api.Brokers.Hydrophones
{
    [ExcludeFromCodeCoverage]
    public partial class HydrophoneBroker : IHydrophoneBroker
    {
        private readonly AppSettings _appSettings;
        private readonly string _apiUrl;

        public HydrophoneBroker(AppSettings appSettings)
        {
            _appSettings = appSettings;
            _apiUrl = _appSettings.HydrophoneFeedUrl;
        }

        public async Task<List<HydrophoneData>> GetFeedsAsync()
        {
            using HttpClient client = new();

            // Send a GET request to the API endpoint
            HttpResponseMessage response = await client.GetAsync(_apiUrl);

            // Ensure a successful response
            response.EnsureSuccessStatusCode();

            // Read the JSON response as a string
            string json = await response.Content.ReadAsStringAsync();

            // Deserialize the JSON string into a list of Data objects
            HydrophoneRootObject root = JsonConvert.DeserializeObject<HydrophoneRootObject>(json);

            // Extract and return the 'data' property
            return root?.Data ?? new List<HydrophoneData>();
        }
    }
}

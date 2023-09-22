namespace OrcaHello.Web.UI.Brokers
{
    public partial class DetectionAPIBroker
    {
        private const string hydrophoneRelativeUrl = "hydrophones";

        public async ValueTask<HydrophoneListResponse> GetAllHydrophonesAsync() =>
            await this.GetAsync<HydrophoneListResponse>(hydrophoneRelativeUrl);
    }
}

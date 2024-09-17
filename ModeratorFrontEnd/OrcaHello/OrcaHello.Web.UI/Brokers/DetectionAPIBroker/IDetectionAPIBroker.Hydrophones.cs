namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<HydrophoneListResponse> GetAllHydrophonesAsync();
    }
}

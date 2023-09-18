namespace OrcaHello.Web.Api.Brokers.Hydrophones
{
    public partial interface IHydrophoneBroker
    {
        Task<List<HydrophoneData>> GetFeedsAsync();
    }
}

namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<TagListResponse> GetAllTagsAsync();
        ValueTask<TagListForTimeframeResponse> GetFilteredTagsAsync(string queryString);
    }
}

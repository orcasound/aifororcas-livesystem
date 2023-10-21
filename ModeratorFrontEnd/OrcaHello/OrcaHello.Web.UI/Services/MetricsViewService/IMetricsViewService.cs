namespace OrcaHello.Web.UI.Services
{
    public interface IMetricsViewService
    {
        ValueTask<List<string>> RetrieveFilteredTagsAsync(TagFilter options);
    }
}

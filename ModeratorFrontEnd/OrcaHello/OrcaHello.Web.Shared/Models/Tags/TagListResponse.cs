namespace OrcaHello.Web.Shared.Models.Tags
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A list of tags")]
    public class TagListResponse
    {
        [SwaggerSchema("The list of tags in ascending order")]
        public List<string> Tags { get; set; } = new List<string>();
        [SwaggerSchema("The total number of tags in the list")]
        public int Count { get; set; }
    }

    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A list of tags for the given timeframe")]
    public class TagListForTimeframeResponse : TagListResponse
    {
        public DateTime FromDate { get; set; }
        [SwaggerSchema("The ending date of the timeframe")]
        public DateTime ToDate { get; set; }
    }
}

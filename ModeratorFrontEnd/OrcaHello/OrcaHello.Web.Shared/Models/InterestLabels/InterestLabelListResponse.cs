namespace OrcaHello.Web.Shared.Models.InterestLabels
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A list of unique interest labels.")]
    public class InterestLabelListResponse
    {
        [SwaggerSchema("The list of interest labels in ascending order")]
        public List<string> InterestLabels { get; set; } = new List<string>();
        [SwaggerSchema("The total number of interst labels in the list")]
        public int Count { get; set; }
    }
}

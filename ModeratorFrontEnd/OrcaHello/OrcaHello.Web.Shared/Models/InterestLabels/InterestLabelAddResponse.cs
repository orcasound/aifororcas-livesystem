namespace OrcaHello.Web.Shared.Models.InterestLabels
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The results of an interest label add action.")]
    public class InterestLabelAddResponse
    {
        [SwaggerSchema("The id of the detection updated.")]
        public string Id { get; set; }

        [SwaggerSchema("The interest label that was added.")]
        public string LabelAdded { get; set; }

    }
}

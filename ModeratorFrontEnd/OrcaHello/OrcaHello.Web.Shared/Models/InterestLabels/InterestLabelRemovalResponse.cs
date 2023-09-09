namespace OrcaHello.Web.Shared.Models.InterestLabels
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The results of an interest label removal action.")]
    public class InterestLabelRemovalResponse
    {
        [SwaggerSchema("The id of the detection updated.")]
        public string Id { get; set; }

        [SwaggerSchema("The interest label that was removed.")]
        public string LabelRemoved { get; set; }

    }
}

namespace OrcaHello.Web.Shared.Models.Moderators
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The State metrics for the given timeframe and moderator.")]
    public class MetricsForModeratorResponse : MetricsResponseBase
    {
        [SwaggerSchema("The name of the moderator.")]
        public string Moderator { get; set; }
    }
}

namespace OrcaHello.Web.Shared.Models.Metrics
{
    [ExcludeFromCodeCoverage]
    public class MetricsResponseBase
    {
        [SwaggerSchema("The number of reviewed detections with no whale call.")]
        public int Negative { get; set; }
        [SwaggerSchema("The number of reviewed detections with confirmed whale call.")]
        public int Positive { get; set; }
        [SwaggerSchema("The number of reviewed detections where whale call could not be determined.")]
        public int Unknown { get; set; }

        [SwaggerSchema("The starting date of the timeframe.")]
        public DateTime FromDate { get; set; }

        [SwaggerSchema("The ending date of the timeframe.")]
        public DateTime ToDate { get; set; }
    }

    [ExcludeFromCodeCoverage]
    [SwaggerSchema("The State metrics for the given timeframe.")]
    public class MetricsResponse : MetricsResponseBase
    {
        [SwaggerSchema("The number of unreviewed detections.")]
        public int Unreviewed { get; set; }
    }
}

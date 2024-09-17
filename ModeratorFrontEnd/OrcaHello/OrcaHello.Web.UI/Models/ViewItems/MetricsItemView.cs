namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class MetricsItemView
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; } = 0;
        public string Color { get; set; } = string.Empty;

        public MetricsItemView(string name, int value, string color)
        {
            Name = name;
            Value = value;
            Color = color;
        }

        public static Func<MetricsResponse, MetricsItemViewResponse> AsMetricsItemViewResponse =>
            metricsResponse => new MetricsItemViewResponse
            {
                MetricsItemViews = new List<MetricsItemView>()
                {
                    new(DetectionState.Unreviewed.ToString(), metricsResponse.Unreviewed, "#219696"),
                    new(DetectionState.Positive.ToString(), metricsResponse.Positive, "#468f57"),
                    new(DetectionState.Negative.ToString(), metricsResponse.Negative, "#bb595f"),
                    new(DetectionState.Unknown.ToString(), metricsResponse.Unknown, "#bc913e")
                },
                FromDate = metricsResponse.FromDate,
                ToDate = metricsResponse.ToDate
            };
    }
}

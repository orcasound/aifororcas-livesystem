namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class DetectionItemViewResponse
    {
        public int Count { get; set; }
        public List<DetectionItemView> DetectionItemViews { get; set; } = null!;
    }
}

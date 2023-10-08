namespace OrcaHello.Web.UI.Models
{
    public class DetectionItemView
    {
        public string Id { get; set; }
        public string AudioUri { get; set; }
        public string SpectrogramUri { get; set; }
        public string State { get; set; }
        public string LocationName { get; set; }
        public decimal Confidence { get; set; }
        public LocationItemView Location { get; set; }
        public List<AnnotationItemView> Annotations { get; set; }
        public DateTime Timestamp { get; set; }
        public string Comments { get; set; }
        public string Moderator { get; set; }
        public DateTime? Moderated { get; set; }
        public string Tags { get; set; }
        public string InterestLabel { get; set; }
        public string AverageConfidence { get => $"{Confidence.ToString("00.##")}% average confidence"; }
        public string SmallConfidence { get => $"{Confidence.ToString("F2")}%"; }
        public string DetectionCount { get => $"{Annotations.Count} detections"; }
    }
}
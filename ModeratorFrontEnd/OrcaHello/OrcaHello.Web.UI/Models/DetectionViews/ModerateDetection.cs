namespace OrcaHello.Web.UI.Models.DetectionViews
{
    public class ModerateDetection
    {
        public string Id { get; set; }
        public string State { get; set; }
        public string Comments { get; set; }
        public string Moderator { get; set; }
        public DateTime? Moderated { get; set; }
        public string Tags { get; set; }
        public string InterestLabel { get; set; }
    }
}

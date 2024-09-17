namespace OrcaHello.Console.DataMigration.Models
{
    public class Metadata2
    {
        public string id { get; set; }
        public string state { get; set; }
        public string locationName { get; set; }
        public string audioUri { get; set; }
        public string imageUri { get; set; }
        public DateTime timestamp { get; set; }
        public decimal whaleFoundConfidence { get; set; }
        public Location location { get; set; }
        public List<Prediction> predictions { get; set; } = new List<Prediction>();
        public string comments { get; set; }
        public string dateModerated { get; set; }
        public string moderator { get; set; }
        public List<string> tags { get; set; } = new List<string>();
    }
}

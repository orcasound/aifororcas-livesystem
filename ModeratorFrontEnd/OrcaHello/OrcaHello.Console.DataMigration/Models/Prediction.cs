namespace OrcaHello.Console.DataMigration.Models
{
    public class Prediction
    {
        public int id { get; set; }
        public decimal startTime { get; set; }
        public decimal duration { get; set; }
        public decimal confidence { get; set; }
    }
}

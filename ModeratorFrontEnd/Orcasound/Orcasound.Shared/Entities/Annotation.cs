namespace Orcasound.Shared.Entities
{
	public class Annotation
	{
		public int Id { get; set; }
		public decimal StartTime { get; set; }
		public decimal Duration { get; set; }
		public decimal Confidence { get; set; }
	}
}

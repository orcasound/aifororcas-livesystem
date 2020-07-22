namespace ModeratorCandidates.Shared.Models
{
	public class AIAnnotation
	{
		public int id { get; set; }
		public decimal startTime { get; set; }
		public decimal duration { get; set; }
		public decimal confidence { get; set; }
	}
}

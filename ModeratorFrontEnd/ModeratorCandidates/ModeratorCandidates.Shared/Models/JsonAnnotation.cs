namespace ModeratorCandidates.Shared.Models
{
	// Annotations are a collection of potential hits within the given Import clip
	public class JsonAnnotation
	{
		public decimal start_time_s {get; set;}
		public decimal duration_s { get; set; }
		public decimal confidence { get; set; }
	}
}

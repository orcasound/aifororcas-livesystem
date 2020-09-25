namespace Orcasound.Shared.Entities
{
	public class Pagination
	{
		public int Page { get; set; } = 1;
		public int RecordsPerPage { get; set; } = 10;
		public string SortBy { get; set; } = "confidence";
		public string SortOrder { get; set; } = "desc";
		public string Timeframe { get; set; } = "all";

	}
}

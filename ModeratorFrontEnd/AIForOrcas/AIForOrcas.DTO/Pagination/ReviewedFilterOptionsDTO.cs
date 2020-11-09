namespace AIForOrcas.DTO
{
	public class ReviewedFilterOptionsDTO : IFilterOptions
	{
		public string SortOrder { get; set; }
		public string SortBy { get; set; }
		public string Timeframe { get; set; }

		public string QueryString { get => $"sortBy={SortBy}&sortOrder={SortOrder}&timeframe={Timeframe}"; }
	}
}

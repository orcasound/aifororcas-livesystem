namespace AIForOrcas.DTO.API
{
	/// <summary>
	/// Query parameters to present to the detections endpoint.
	/// </summary>
	public class DetectionQueryParameters
	{
		/// <summary>
		/// Page number to retrieve.
		/// </summary>
		/// <example>1</example>
		public int Page { get; set; } = 1;

		/// <summary>
		/// Property to sort by (confidence, timestamp).
		/// </summary>
		/// <example>timestamp</example>
		public string SortBy { get; set; } = "timestamp";

		/// <summary>
		/// Order in which to sort the results (asc, desc).
		/// </summary>
		/// <example>desc</example>
		public string SortOrder { get; set; } = "desc";

		/// <summary>
		/// Timeframe for the record set (last 30m, 24h, 1w, 30d, all).
		/// </summary>
		/// <example>all</example>
		public string Timeframe { get; set; } = "all";

		/// <summary>
		/// Number of records per page to retrieve.
		/// </summary>
		/// <example>5</example>
		public int RecordsPerPage
		{
			get => _recordsPerPage;
			set
			{
				_recordsPerPage = (value > _maxRecordsPerPage) ? _maxRecordsPerPage : value;
			}
		}

		private int _recordsPerPage = 10;
		private readonly int _maxRecordsPerPage = 50;
	}
}

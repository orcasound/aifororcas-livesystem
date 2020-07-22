namespace ModeratorCandidates.Shared.Models
{
	public class Pagination
	{
		public int Page { get; set; } = 1;

		private int recordsPerPage = 10;

		private readonly int maxRecordsPerPage = 50;

		public int RecordsPerPage
		{
			get
			{
				return recordsPerPage;
			}
			set
			{
				recordsPerPage = (value > maxRecordsPerPage) ? maxRecordsPerPage : value;
			}
		}
	}
}

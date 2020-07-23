namespace Orcasound.Shared.Entities
{
	public class PaginatedResponse<T>
	{
		public T Response { get; set; }
		public int TotalAmountPages { get; set; }
		public int TotalNumberRecords { get; set; }
	}
}

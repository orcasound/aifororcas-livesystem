namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class TagStateView
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public bool IsExpanded { get; set; } = false;
        public bool IsLoading { get; set; } = false;
        public List<string> Items { get; set; } = null!;
        public int Count { get; set; }

        public void Reset(DateTime fromDate, DateTime toDate)
        {
            IsExpanded = false;
            IsLoading = false;
            Items = null!;
            Count = 0;
            FromDate = fromDate;
            ToDate = toDate;
        }

        public void Close()
        {
            IsExpanded = false;
            IsLoading = false;
            Items = null!;
            Count = 0;
        }

        public void Toggle()
        {
            IsExpanded = !IsExpanded;

            if (IsExpanded && Items == null)
            {
                Items = new();
            }
        }

        public bool IsPopulated => Items != null && Items.Any();

        public bool IsEmpty => Items != null && !Items.Any();
    }
}

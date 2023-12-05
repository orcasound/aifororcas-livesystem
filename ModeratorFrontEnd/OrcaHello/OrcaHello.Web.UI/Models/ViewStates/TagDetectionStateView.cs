namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class TagDetectionStateView
    {
        public string Tag { get; set; } = null!;
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public int Page { get; set; } = 1;
        public int PageSize { get; set; } = 10;
        public bool IsExpanded { get; set; } = false;
        public bool IsLoading { get; set; } = false;
        public List<DetectionItemView> Items { get; set; } = null!;
        public int Count { get; set; }

        public void Reset(DateTime fromDate, DateTime toDate)
        {
            Page = 1;
            IsExpanded = false;
            IsLoading = false;
            Items = new();
            Count = 0;
            FromDate = fromDate;
            ToDate = toDate;
            Tag = string.Empty;
        }

        public void SelectedTagReset(string tag)
        {
            Tag = tag;
            Count = 0;
            Page = 1;
            IsExpanded = false;
            IsLoading = false;
            Items = new();
        }

        public bool IsPopulated => Items != null && Items.Any();
    }
}

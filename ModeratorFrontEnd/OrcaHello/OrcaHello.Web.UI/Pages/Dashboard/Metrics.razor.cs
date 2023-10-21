using OrcaHello.Web.UI.Models.Dashboard;
using System;

namespace OrcaHello.Web.UI.Pages.Dashboard
{
    public partial class Metrics
    {
        [Inject]
        public IMetricsViewService ViewService { get; set; } = null!;

        protected bool IsLoading = false;
        protected bool IsLoadingTags = false;
        protected List<string> TagsForTimeframe = new();

        // Timeframe
        protected List<DropdownOption> TimeframeDropdownOptions = new()
        {
            new(Timeframe.ThirtyMinutes, "Last 30 Minutes"),
            new(Timeframe.TwentyFourHours, "Last 24 hours"),
            new(Timeframe.SevenDays, "Last Week"),
            new(Timeframe.ThirtyDays, "Last Month"),
            new(Timeframe.ThreeMonths, "Last 3 Months"),
            new(Timeframe.SixMonths, "Last 6 Months"),
            new(Timeframe.OneYear, "Last Year"),
            new(Timeframe.All, "All"),
            new(Timeframe.Range, "Select Date Range")
        };

        protected Timeframe SelectedTimeframe;

        // Datetime Range

        protected DateTime? SelectedStartDateTime = null;
        protected DateTime? SelectedEndDateTime = null;

        protected List<string> FillColors = new();

        protected MetricsItemViewResponse MetricsItemViewResponse { get; set; } = null!;

        protected override async Task OnInitializedAsync()
        {
            MetricsItemViewResponse = new MetricsItemViewResponse
            {
                MetricsItemViews = new()
                {
                    new MetricsItemView("Unreviewed", 1958),
                    new MetricsItemView("Negative", 3470),
                    new MetricsItemView("Positive", 600),
                    new MetricsItemView("Unknown", 30)
                }
            };

            FillColors = new() { "#219696", "#bb595f", "#468f57", "#bc913e" };
        }


        protected void OnTimeframeChanged()
        {
            if (SelectedTimeframe != Timeframe.Range)
            {
                SelectedStartDateTime = null;
                SelectedEndDateTime = null;
            }
            else if (SelectedStartDateTime == null || SelectedEndDateTime == null)
            {
                SelectedStartDateTime = DateTime.UtcNow.AddDays(-1);
                SelectedEndDateTime = DateTime.UtcNow;
            }
        }

        protected void OnApplyFilterClicked()
        {

        }

        protected async Task OnTagsAccordionExpanded(object value)
        {
            IsLoadingTags = true;

            var TagFilter = new TagFilter
            {
                FromDate = DateTime.UtcNow.AddDays(-365),
                ToDate = DateTime.UtcNow
            };

            TagsForTimeframe = await ViewService.RetrieveFilteredTagsAsync(TagFilter);

            IsLoadingTags = false;

            await InvokeAsync(StateHasChanged);
        }

        protected async Task OnTagsAccordionCollapsed(object value)
        {

        }
    }
}

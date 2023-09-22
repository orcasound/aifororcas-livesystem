namespace OrcaHello.Web.UI.Pages.Components
{
    public partial class PageHeaderAndTitle
    {
        [Parameter]
        public string Title { get; set; }

        [Parameter]
        public string Description { get; set; }

        [Parameter]
        public string BadgeValue { get; set; }
    }
}

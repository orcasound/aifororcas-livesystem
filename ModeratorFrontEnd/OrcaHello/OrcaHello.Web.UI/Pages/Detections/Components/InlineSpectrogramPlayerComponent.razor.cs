namespace OrcaHello.Web.UI.Pages.Detections.Components
{
    public partial class InlineSpectrogramPlayerComponent
    {
        [Parameter]
        public string Id { get; set; } = string.Empty;

        [Parameter]
        public string AudioUri { get; set; } = string.Empty;

        [Parameter]
        public string SpectrogramUri { get; set; } = string.Empty;
    }
}

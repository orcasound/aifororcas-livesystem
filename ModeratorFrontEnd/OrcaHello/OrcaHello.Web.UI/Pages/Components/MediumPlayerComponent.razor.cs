namespace OrcaHello.Web.UI.Pages.Components
{
    [ExcludeFromCodeCoverage]
    public partial class MediumPlayerComponent
    {
        [Parameter]
        public DetectionItemView ItemView { get; set; } = null!;

        protected ElementReference imageRef; // reference to the rendered image
        protected ElementReference audioElement; // reference to the rendered audio

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                await JSRuntime.InvokeVoidAsync("addAudioEventListenersById", ItemView.Id);
            }
        }
    }
}

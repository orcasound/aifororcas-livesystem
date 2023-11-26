namespace OrcaHello.Web.UI.Pages.Components
{
    [ExcludeFromCodeCoverage]
    public partial class SmallMapComponent
    {
        [Parameter]
        public int Id { get; set; } = 0;

        [Parameter]
        public double Longitude { get; set; } = double.MinValue;

        [Parameter]
        public double Latitude { get; set; } = double.MinValue;

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                await JSRuntime.InvokeVoidAsync("LoadSmallBingMap", Id, Latitude, Longitude);
            }
        }

    }
}

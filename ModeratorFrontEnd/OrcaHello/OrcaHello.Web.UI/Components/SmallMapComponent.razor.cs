namespace OrcaHello.Web.UI.Components
{
    public partial class SmallMapComponent
    {
        [Parameter]
        public int Id { get; set; }

        [Parameter]
        public double Longitude { get; set; }

        [Parameter]
        public double Latitude { get; set; }

        protected override async Task OnAfterRenderAsync(bool firstRender)
        {
            if (firstRender)
            {
                await JSRuntime.InvokeVoidAsync("LoadSmallBingMap", Id, Latitude, Longitude);
            }
        }

    }
}

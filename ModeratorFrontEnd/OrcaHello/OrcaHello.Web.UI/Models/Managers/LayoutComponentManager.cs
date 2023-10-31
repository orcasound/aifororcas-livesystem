namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class LayoutComponentManager : LayoutComponentBase
    {
        [Inject]
        protected NavigationManager NavManager { get; set; }

        [Inject]
        public IJSRuntime JSRuntime { get; set; }

        [Inject]
        public DialogService DialogService { get; set; }

        [Inject]
        public NotificationService NotificationService { get; set; }

        [Inject]
        public ContextMenuService ContextMenuService { get; set; }

        [Inject]
        public TooltipService TooltipService { get; set; }

        [Inject]
        public ILoggerFactory LoggerFactory { get; set; }

        public ILogger Logger;

        protected override void OnInitialized()
        {
            Logger = LoggerFactory.CreateLogger(GetType().Name);
            base.OnInitialized();
        }

        public void CloseDialog()
        {
            DialogService.Close();
        }

        public void ShowTooltip(ElementReference elementReference, string message, TooltipOptions options = null) =>
            TooltipService.Open(elementReference, message, options);
    }
}

namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class LayoutComponentManager : LayoutComponentBase
    {
        [Inject]
        protected NavigationManager NavManager { get; set; } = null!;

        [Inject]
        public IJSRuntime JSRuntime { get; set; } = null!;

        [Inject]
        public DialogService DialogService { get; set; } = null!;

        [Inject]
        public NotificationService NotificationService { get; set; } = null!;

        [Inject]
        public ContextMenuService ContextMenuService { get; set; } = null!;

        [Inject]
        public TooltipService TooltipService { get; set; } = null!;

        [Inject]
        public ILoggerFactory LoggerFactory { get; set; } = null!;

        public ILogger Logger = null!;

        protected override void OnInitialized()
        {
            Logger = LoggerFactory.CreateLogger(GetType().Name);
            base.OnInitialized();
        }

        public void CloseDialog()
        {
            DialogService.Close();
        }

        public void ShowTooltip(ElementReference elementReference, string message, TooltipOptions options = null!) =>
            TooltipService.Open(elementReference, message, options);
    }
}

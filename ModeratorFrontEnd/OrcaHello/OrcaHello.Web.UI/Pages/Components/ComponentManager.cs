namespace OrcaHello.Web.UI.Pages.Components
{
    [ExcludeFromCodeCoverage]
    public class ComponentManager : ComponentBase
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
            Logger = LoggerFactory.CreateLogger(this.GetType().Name);
            base.OnInitialized();
        }

        public void CloseDialog()
        {
            DialogService.Close();
        }

        public void ShowTooltip(ElementReference elementReference, string message, TooltipOptions options = null) =>
            TooltipService.Open(elementReference, message, options);

        public void LogAndReportUnknownException(Exception exception)
        {
            var fullMessage = exception.GetAllMessages();
            var fullStackTrace = exception.GetFullStackTrace();

            var message = new NotificationMessage
            {
                Severity = NotificationSeverity.Error,
                Summary = "Unknown Failure",
                Detail = $"{fullMessage} {fullStackTrace} - Please contact support for resolution.",
                Duration = 30000,
                Style = "width: 50vw"
            };

            NotificationService.Notify(message);
            Logger.LogError(exception, "Unknown Failure");
        }
    }
}

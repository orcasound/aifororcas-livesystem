namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class ComponentManager : ComponentBase
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

        public void ReportSuccess(string summary, string detail)
        {
            NotificationService.Notify(
                new NotificationMessage
                {
                    Severity = NotificationSeverity.Success,
                    Summary = summary,
                    Detail = detail,
                    Duration = 5000
                });
        }

        public void ReportError(string summary, string detail)
        {
            NotificationService.Notify(
                new NotificationMessage
                {
                    Severity = NotificationSeverity.Error,
                    Summary = summary,
                    Detail = detail,
                    Duration = 5000
                });
        }
    }
}

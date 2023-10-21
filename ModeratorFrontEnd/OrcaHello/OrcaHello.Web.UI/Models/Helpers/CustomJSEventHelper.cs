namespace OrcaHello.Web.UI.Models.Helpers
{
    public class CustomJSEventHelper
    {
        private readonly Func<EventArgs, Task> _callback;

        public CustomJSEventHelper(Func<EventArgs, Task> callback)
        {
            _callback = callback;
        }

        [JSInvokable]
        public Task OnJSCustomEvent(EventArgs args) => _callback(args);
    }
}

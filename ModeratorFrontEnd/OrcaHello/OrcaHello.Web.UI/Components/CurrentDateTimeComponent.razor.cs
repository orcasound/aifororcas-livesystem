namespace OrcaHello.Web.UI.Components
{
    public partial class CurrentDateTimeComponent
    {
        private System.Timers.Timer _timer;
        private DateTime _currentDateTime;

        protected override void OnInitialized()
        {
            _currentDateTime = DateTime.UtcNow;

            _timer = new System.Timers.Timer(1000); // Set the interval to 1 second
            _timer.Elapsed += OnTimerElapsed;
            _timer.Start();
        }

        private void OnTimerElapsed(object sender, ElapsedEventArgs e)
        {
            _currentDateTime = DateTime.UtcNow;
            InvokeAsync(StateHasChanged); // Invoke StateHasChanged on the UI thread
        }

        public void Dispose()
        {
            _timer?.Dispose();
        }
    }
}

namespace OrcaHello.Web.UI.Shared
{
    public partial class RightTopBar
    {
        [Inject]
        IAccountService AccountService { get; set; } = null!;

        private string DisplayName { get; set; } = string.Empty;

        protected override async Task OnInitializedAsync()
        {
            DisplayName = await AccountService.GetDisplayname();
        }

        private async Task Login()
        {
            await AccountService.Login();
        }

        private async Task OnProfileMenuClicked(RadzenProfileMenuItem item)
        {
            if(item.Text == "Log out")
            {
                bool? result = await DialogService.Confirm("Select \"Log Out\" below if you are ready to end your current session.", "Ready to Leave?", new ConfirmOptions() { OkButtonText = "Log Out", CancelButtonText = "Cancel" });

                if (result == true)
                {
                    await AccountService.Logout();
                }
            }
        }
    }
}

namespace OrcaHello.Web.UI.Shared
{
    public partial class RightTopBar
    {
        [Inject]
        IAccountService AccountService { get; set; }

        private async Task Login()
        {
            await AccountService.Login();
        }
    }
}

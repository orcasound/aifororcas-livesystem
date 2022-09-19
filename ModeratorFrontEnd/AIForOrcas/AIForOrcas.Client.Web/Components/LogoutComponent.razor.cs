namespace AIForOrcas.Client.Web.Components;

public partial class LogoutComponent
{

    [Inject]
    IAccountService AccountService { get; set; }

    private async Task Logout()
    {
        await AccountService.Logout();
    }
}
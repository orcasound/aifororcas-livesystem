namespace OrcaHello.Web.UI.Services
{ 
    public interface IAccountService
    {
        Task Login();
        Task Logout();
        Task LogoutIfExpired();
        Task<string> GetToken();
        Task<string> GetDisplayname();
        Task<string> GetUsername();
    }
}

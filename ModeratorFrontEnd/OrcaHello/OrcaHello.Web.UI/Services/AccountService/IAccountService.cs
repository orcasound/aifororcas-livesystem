namespace OrcaHello.Web.UI.Services
{ 
    public interface IAccountService
    {
        Task Login();
        Task Logout();
        Task<string> GetToken();
        Task<string> GetDisplayname();
        Task<string> GetUsername();
    }
}

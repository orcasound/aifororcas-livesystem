using System.IdentityModel.Tokens.Jwt;

namespace OrcaHello.Web.UI.Services
{
    public class AccountService : IAccountService
    {
        private readonly HttpClient _httpService;
        private readonly AuthenticationStateProvider _authenticationStateProvider;
        private readonly BlazoradeMsalService _msalService;
        private readonly ILocalStorageService _localStorage;
        private readonly AppSettings _appSettings;

        public AccountService(
            HttpClient httpService,
            AuthenticationStateProvider authenticationStateProvider,
            BlazoradeMsalService msalService,
            ILocalStorageService localStorage,
            AppSettings appSettings)
        {
            _httpService = httpService;
            _authenticationStateProvider = authenticationStateProvider;
            _msalService = msalService;
            _localStorage = localStorage;
            _appSettings = appSettings;
        }

        public async Task<string> GetToken()
        {
            string savedToken = await _localStorage.GetItemAsync<string>("authToken");
            return savedToken;
        }

        public async Task<string> GetDisplayname()
        {
            if (_authenticationStateProvider != null)
            {
                AuthenticationState authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
                ClaimsPrincipal user = authState.User;

                if (user?.Identity != null && user.Identity.IsAuthenticated)
                {
                    var name = user.FindFirst(c => c.Type == "name")?.Value;
                    var identity = user.Identity.Name;
                    return !string.IsNullOrWhiteSpace(name) ? name : identity ?? string.Empty;
                }
            }

            return string.Empty;
        }

        public async Task<string> GetUsername()
        {
            if (_authenticationStateProvider != null)
            {
                AuthenticationState authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
                ClaimsPrincipal user = authState.User;

                if (user?.Identity != null && user.Identity.IsAuthenticated)
                {
                    var username = user.FindFirst(c => c.Type == "email")?.Value;
                    var identity = user.Identity.Name;
                    return !string.IsNullOrWhiteSpace(username) ? username : identity ?? string.Empty;
                }
            }

            return string.Empty;
        }

        public async Task Login()
        {
            AuthenticationResult token = null!;

            var scopes = new string[] { $"api://{_appSettings.AzureAd.ClientId}/{_appSettings.AzureAd.DefaultScope}" };

            try
            {
                token = await _msalService.AcquireTokenAsync(prompt: LoginPrompt.Login, scopes: scopes);

                token.ExpiresOn = DateTime.UtcNow.AddSeconds(30);

                await _localStorage.SetItemAsync("authToken", token.AccessToken);

                await ((ApiAuthenticationStateProvider)_authenticationStateProvider).MarkUserAsAuthenticated();

                _httpService.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("bearer", token.AccessToken);
            }
            catch (Exception exception)
            {
                // TODO: Better reporting for this
                Console.WriteLine(exception.Message);
            }
        }

        public async Task Logout()
        {
            await _localStorage.ClearAsync();
            ((ApiAuthenticationStateProvider)_authenticationStateProvider).MarkUserAsLoggedOut();
            _httpService.DefaultRequestHeaders.Clear();
        }

        public async Task LogoutIfExpired()
        {
            AuthenticationState authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
            ClaimsPrincipal user = authState.User;

            if (user?.Identity != null && user.Identity.IsAuthenticated)
            {
                var savedToken = await _localStorage.GetItemAsync<string>("authToken");

                if (((ApiAuthenticationStateProvider)_authenticationStateProvider).IsTokenExpired(savedToken))
                {
                    await _localStorage.ClearAsync();
                    ((ApiAuthenticationStateProvider)_authenticationStateProvider).MarkUserAsLoggedOut();
                    _httpService.DefaultRequestHeaders.Clear();
                }
            }
        }
    }
}
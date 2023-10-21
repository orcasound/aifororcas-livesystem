namespace OrcaHello.Web.UI.Services
{
    public class AccountService : IAccountService
    {
        private readonly HttpClient _httpService;
        private readonly BlazoradeMsalService _msalService;
        private readonly AppSettings _appSettings;
        private const string _tokenName = "authToken";
        private const string _headerName = "bearer";
        private readonly ApiAuthenticationStateProvider _apiAuthenticationStateProvider;
        private readonly NavigationManager _navigationManager;

        public AccountService(
            HttpClient httpService,
            ApiAuthenticationStateProvider apiAuthenticationStateProvider,
            BlazoradeMsalService msalService,
            NavigationManager navigationManager,
            AppSettings appSettings)
        {
            _httpService = httpService;
            _apiAuthenticationStateProvider = apiAuthenticationStateProvider;
            _msalService = msalService;
            _navigationManager = navigationManager;
            _appSettings = appSettings;
        }

        public async Task<string> GetDisplayName()
        {
            if (_apiAuthenticationStateProvider != null)
            {
                AuthenticationState authState = await _apiAuthenticationStateProvider.GetAuthenticationStateAsync();
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

        public async Task<string> GetUserName()
        {
            if (_apiAuthenticationStateProvider != null)
            {
                AuthenticationState authState = await _apiAuthenticationStateProvider.GetAuthenticationStateAsync();
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

                await _apiAuthenticationStateProvider.SetToken(token.AccessToken);

                await _apiAuthenticationStateProvider.MarkUserAsAuthenticated();

                _httpService.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue(_headerName, token.AccessToken);
            }
            catch (Exception exception)
            {
                // TODO: Better reporting for this
                Console.WriteLine(exception.Message);
            }
        }

        public async Task Logout()
        {
            await _apiAuthenticationStateProvider.ClearToken();

            _apiAuthenticationStateProvider.MarkUserAsLoggedOut();

            _httpService.DefaultRequestHeaders.Clear();
        }

        public async Task LogoutIfExpired()
        {
            try
            {
                AuthenticationState authState = await _apiAuthenticationStateProvider.GetAuthenticationStateAsync();
                ClaimsPrincipal user = authState.User;

                if (user?.Identity != null && user.Identity.IsAuthenticated)
                {
                    var savedToken = await _apiAuthenticationStateProvider.GetToken();

                    if (!string.IsNullOrWhiteSpace(savedToken) && _apiAuthenticationStateProvider.IsTokenExpired(savedToken))
                    {
                        await _apiAuthenticationStateProvider.ClearToken();

                        _apiAuthenticationStateProvider.MarkUserAsLoggedOut();
                        _httpService.DefaultRequestHeaders.Clear();
                        _navigationManager.NavigateTo("/");

                    }
                }
            }
            catch 
            {
                _apiAuthenticationStateProvider.ClearMemoryToken();

                _apiAuthenticationStateProvider.MarkUserAsLoggedOut();
                _httpService.DefaultRequestHeaders.Clear();
                _navigationManager.NavigateTo("/");
            }
        }
    }
}
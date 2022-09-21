﻿namespace AIForOrcas.Client.Web.Services;

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
        var savedToken = await _localStorage.GetItemAsync<string>("authToken");
        return savedToken;
    }

    public async Task<string> GetDisplayname()
    {
        if (_authenticationStateProvider != null)
        {
            var authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
            var user = authState.User;

            if (user.Identity.IsAuthenticated)
            {
                var name = user.FindFirst(c => c.Type == "name")?.Value;
                var identity = user.Identity.Name;
                return string.IsNullOrWhiteSpace(name) ? identity : name;
            }
        }

        return string.Empty;
    }

    public async Task<string> GetUsername()
    {
        if (_authenticationStateProvider != null)
        {
            var authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
            var user = authState.User;

            if (user.Identity.IsAuthenticated)
            {
                var username = user.FindFirst(c => c.Type == "email")?.Value;
                var identity = user.Identity.Name;
                return string.IsNullOrWhiteSpace(username) ? identity : username;
            }
        }

        return string.Empty;
    }

    public async Task Login()
    {
        AuthenticationResult token = null;

        var scopes = new string[] { $"api://{_appSettings.AzureAd.ClientId}/{_appSettings.AzureAd.DefaultScope}" };

        try
        {
            token = await _msalService.AcquireTokenAsync(prompt: LoginPrompt.Login, scopes: scopes);

            await _localStorage.SetItemAsync("authToken", token.AccessToken);

            await ((ApiAuthenticationStateProvider)_authenticationStateProvider).MarkUserAsAuthenticated();

            _httpService.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("bearer", token.AccessToken);
        }
        catch (Exception exception)
        {
            Console.WriteLine(exception.Message);
        }
    }

    public async Task Logout()
    {
        await _localStorage.ClearAsync();
        ((ApiAuthenticationStateProvider)_authenticationStateProvider).MarkUserAsLoggedOut();
        _httpService.DefaultRequestHeaders.Clear();
    }
}

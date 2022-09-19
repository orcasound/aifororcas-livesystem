namespace AIForOrcas.Client.Web.Extensions;

public static class External
{
    public static void ConfigureExternalUtilities(this WebApplicationBuilder builder, AppSettings appSettings)
    {
        builder.Services.AddBlazoredLocalStorage();
        builder.Services.AddBlazoredToast();

        builder.Services.AddBlazoradeMsal((sp, o) =>
        {
            o.ClientId = appSettings.AzureAd.ClientId;
            o.TenantId = appSettings.AzureAd.TenantId;
            o.DefaultScopes = new string[] { appSettings.AzureAd.DefaultScope };
            o.InteractiveLoginMode = InteractiveLoginMode.Popup;
            o.TokenCacheScope = TokenCacheScope.Persistent;
        });
    }
}

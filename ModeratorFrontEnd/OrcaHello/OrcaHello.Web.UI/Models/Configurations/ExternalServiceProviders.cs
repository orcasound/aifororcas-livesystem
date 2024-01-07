namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public static class ExternalServiceProviders
    {
        public static void ConfigureExternalUtilities(this WebApplicationBuilder builder, AppSettings appSettings)
        {
            // Blazorade.MSAL

            builder.Services.AddBlazoradeMsal((sp, o) =>
            {
                o.ClientId = appSettings.AzureAd.ClientId;
                o.TenantId = appSettings.AzureAd.TenantId;
                o.DefaultScopes = new string[] { appSettings.AzureAd.DefaultScope };
                o.InteractiveLoginMode = InteractiveLoginMode.Popup;
                o.TokenCacheScope = TokenCacheScope.Persistent;
            });

            // InMemory Storage

            builder.Services.AddMemoryCache();

            // Radzen.Blazor

            builder.Services.AddRadzenComponents();
        }
    }

}

﻿namespace OrcaHello.Web.UI.Models.Extensions
{
    public static class ExternalServices
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

            // Blazored.LocalStorage

            builder.Services.AddBlazoredLocalStorage();

            // Radzen.Blazor

            builder.Services.AddRadzenComponents();
        }
    }

}
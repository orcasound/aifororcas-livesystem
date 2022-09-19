namespace AIForOrcas.Client.Web.Extensions
{
    public static class Authentication
    {
        // Set ModeratorPolicy
        public static void ConfigureModeratorPolicy(this WebApplicationBuilder builder, AppSettings appSettings)
        {
            if (!string.IsNullOrWhiteSpace(appSettings?.AzureAd?.AzureAdGroups?.ModeratorGroupId))
            {
                builder.Services.AddAuthorization(options =>
                {
                    options.AddPolicy("ModeratorRole", policyBuilder =>
                        policyBuilder.RequireClaim("groups", appSettings.AzureAd.AzureAdGroups.ModeratorGroupId));
                });
            }
        }

        // Inject service providers
        public static void ConfigureAuthProviders(this WebApplicationBuilder builder)
        {
            builder.Services.AddScoped<AuthenticationStateProvider, ApiAuthenticationStateProvider>();
            builder.Services.AddScoped<IAccountService, AccountService>();
        }
    }
}

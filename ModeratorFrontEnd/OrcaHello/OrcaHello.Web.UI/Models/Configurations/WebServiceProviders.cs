namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public static class WebServiceProviders
    {
        public static void ConfigureHttpServices(this WebApplicationBuilder builder)
        {
            builder.Services.AddHttpClient();
            builder.Services.AddTransient<IHttpService, HttpService>();
        }

        public static void ConfigureApiClient(this WebApplicationBuilder builder, AppSettings appSettings)
        {
            if (!string.IsNullOrWhiteSpace(appSettings?.APIUrl))
            {
                builder.Services.AddScoped(sp =>
                {
                    var client = new HttpClient();
                    client.BaseAddress = new System.Uri(appSettings.APIUrl);
                    return client;
                });
            }
        }
    }
}

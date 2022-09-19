namespace AIForOrcas.Client.Web.Extensions;

public static class Services
{
    public static void ConfigureDataServices(this WebApplicationBuilder builder)
    {
        builder.Services.AddTransient<IDetectionService, DetectionService>();
        builder.Services.AddTransient<IMetricsService, MetricsService>();
        builder.Services.AddTransient<ITagService, TagService>();
    }

    public static void ConfigureWebServices(this WebApplicationBuilder builder, AppSettings appSettings)
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

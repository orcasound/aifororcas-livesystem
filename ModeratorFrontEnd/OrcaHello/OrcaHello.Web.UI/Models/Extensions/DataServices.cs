namespace OrcaHello.Web.UI.Models.Extensions
{
    public static class DataServices
    {
        public static void ConfigureDataServices(this WebApplicationBuilder builder)
        {
            // Brokers
            builder.Services.AddScoped<IDetectionAPIBroker, DetectionAPIBroker>();

            // Foundation services
            builder.Services.AddScoped<IHydrophoneService, HydrophoneService>();
            builder.Services.AddScoped<IDetectionService, DetectionService>();
            builder.Services.AddScoped<ITagService, TagService>();

            // View services
            builder.Services.AddScoped<IHydrophoneViewService, HydrophoneViewService>();
            builder.Services.AddScoped<IDetectionViewService, DetectionViewService>();
            builder.Services.AddScoped<IMetricsViewService, MetricsViewService>();
        }
    }
}

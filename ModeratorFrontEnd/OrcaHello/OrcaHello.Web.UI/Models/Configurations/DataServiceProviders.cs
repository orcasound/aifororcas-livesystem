namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public static class DataServiceProviders
    {
        public static void ConfigureDataServices(this WebApplicationBuilder builder)
        {
            // Brokers
            builder.Services.AddScoped<IDetectionAPIBroker, DetectionAPIBroker>();

            // Foundation services
            builder.Services.AddScoped<IHydrophoneService, HydrophoneService>();
            builder.Services.AddScoped<IDetectionService, DetectionService>();
            builder.Services.AddScoped<ITagService, TagService>();
            builder.Services.AddScoped<IMetricsService, MetricsService>();
            builder.Services.AddScoped<ICommentService, CommentService>();
            builder.Services.AddScoped<IModeratorService, ModeratorService>();

            // View services
            builder.Services.AddScoped<IHydrophoneViewService, HydrophoneViewService>();
            builder.Services.AddScoped<IDetectionViewService, DetectionViewService>();
            builder.Services.AddScoped<IDashboardViewService, DashboardViewService>();
            builder.Services.AddScoped<ITagViewService, TagViewService>();
        }
    }
}

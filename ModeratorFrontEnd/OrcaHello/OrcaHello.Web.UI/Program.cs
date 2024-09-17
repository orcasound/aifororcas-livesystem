[ExcludeFromCodeCoverage]
internal class Program
{
    private static async Task Main(string[] args)
    {

        var builder = WebApplication.CreateBuilder(args);

        // Inject application 
        AppSettings appSettings = new();
        builder.Configuration.GetSection("AppSettings").Bind(appSettings);
        builder.Services.AddSingleton<AppSettings>(appSettings);

        // Inject external libraries (Models\Extensions\ExternalServices)
        builder.ConfigureExternalUtilities(appSettings);

        builder.Services.AddRazorPages();
        builder.Services.AddServerSideBlazor();

        // Inject authentication services and access policies (Models\Extensions\AuthenticationServices)
        builder.ConfigureAuthProviders();
        builder.ConfigureModeratorPolicy(appSettings);

        // Inject web services (Models\Extensions\WebServices)
        builder.ConfigureHttpServices();
        builder.ConfigureApiClient(appSettings);

        // Inject data services (Modeles\Extensions\DataServices)
        builder.ConfigureDataServices();

        var app = builder.Build();

        // Want to load the hydrophone location names and stick them in an AppSettings singleton so we don't have
        // to look them up each time

        using (var scope = app.Services.CreateScope())
        {
            var hydrophoneService = scope.ServiceProvider.GetRequiredService<IHydrophoneService>();
            var logger = scope.ServiceProvider.GetRequiredService<ILogger<Program>>();

            try
            {
                appSettings.HydrophoneLocationNames = (await hydrophoneService.RetrieveAllHydrophonesAsync())
                    .Select(x => x.Name).ToList();
            }
            catch (Exception ex)
            {
                appSettings.HydrophoneLocationNames = new List<string>();
                // Log the exception as an error with a custom message and the stack trace
                logger.LogError(ex, "Failed to retrieve hydrophone names.");
            }
        }

        if (app.Environment.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }
        else
        {
            app.UseExceptionHandler("/Error");
            // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
            app.UseHsts();
        }

        app.UseHttpsRedirection();
        app.UseStaticFiles();

        app.UseRouting();

        app.MapControllers();
        app.MapBlazorHub();

        app.MapFallbackToPage("/Components/_Host");

        app.Run();

    }
}
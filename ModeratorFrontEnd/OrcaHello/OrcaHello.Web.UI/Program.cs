var builder = WebApplication.CreateBuilder(args);

// Add AppSettings

var appSettings = new AppSettings();
builder.Configuration.GetSection("AppSettings").Bind(appSettings);
builder.Services.AddSingleton(appSettings);

// Add services to the container.
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();

builder.Services.AddRadzenComponents();

builder.Services.AddHttpClient();

builder.Services.AddSingleton<IHttpService, HttpService>();

builder.Services.AddScoped<IDetectionAPIBroker, DetectionAPIBroker>();

builder.Services.AddScoped<IHydrophoneService, HydrophoneService>();
builder.Services.AddScoped<IDetectionService, DetectionService>();

builder.Services.AddScoped<IHydrophoneViewService, HydrophoneViewService>();
builder.Services.AddScoped<IDetectionViewService, DetectionViewService>();

var app = builder.Build();

// Want to load the hydrophone location names and stick them in an AppSettings singleton so we don't have
// to look them up each time

using (var scope = app.Services.CreateScope())
{
    var hydrophoneService = scope.ServiceProvider.GetRequiredService<IHydrophoneService>();
    appSettings.HydrophoneLocationNames = (await hydrophoneService.RetrieveAllHydrophonesAsync()).Select(x => x.Name).ToList();
}

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Components/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/Components/_Host");

app.Run();


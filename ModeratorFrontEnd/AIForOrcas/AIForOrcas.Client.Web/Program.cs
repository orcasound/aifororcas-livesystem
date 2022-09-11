var builder = WebApplication.CreateBuilder(args);

builder.Services.AddAuthentication(AzureADDefaults.AuthenticationScheme)
    .AddAzureAD(options => builder.Configuration.Bind("AzureAd", options));

builder.Services.AddControllersWithViews();

builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("ModeratorRole", policyBuilder =>
        policyBuilder.RequireClaim("groups", builder.Configuration["AzureADGroup:ModeratorGroupId"]));
});

builder.Services.AddBlazoredToast();

builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();

builder.Services.AddHttpContextAccessor();

builder.Services.AddHttpClient<IDetectionService, DetectionService>(client =>
{
    client.BaseAddress = new System.Uri(builder.Configuration["APIUrl"]);
});

builder.Services.AddHttpClient<IMetricsService, MetricsService>(client =>
{
    client.BaseAddress = new System.Uri(builder.Configuration["APIUrl"]);
});

builder.Services.AddHttpClient<ITagService, TagService>(client =>
{
    client.BaseAddress = new System.Uri(builder.Configuration["APIUrl"]);
});

builder.Services.AddScoped<IdentityHelper>();

var app = builder.Build();

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

app.UseAuthentication();
app.UseAuthorization();

app.UseEndpoints(endpoints =>
{
    // MapControllers in needed to enable authentication/authorization through Azure AD
    endpoints.MapControllers();
    endpoints.MapBlazorHub();
    endpoints.MapFallbackToPage("/_Host");
});

app.Run();
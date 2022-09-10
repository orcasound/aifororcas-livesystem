using AIForOrcas.Server.BL.Context;
using AIForOrcas.Server.BL.Services;
using AIForOrcas.Server.Extensions;
using AIForOrcas.Server.Models.Settings;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System;

var builder = WebApplication.CreateBuilder(args);

var appSettings = new AppSettings();
builder.Configuration.GetSection("AppSettings").Bind(appSettings);
builder.Services.AddSingleton<AppSettings>(appSettings);

// Add authentication/authorization middleware (AuthMiddleWareExtensions)
builder.ConfigureCors();
builder.ConfigureJwtAuthentication(builder.Configuration.GetSection("AppSettings:AzureAd"));
builder.ConfigureModeratorPolicy(appSettings);
builder.ConfigureSwagger(appSettings);

// Add needed dependency injections
builder.Services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseCosmos(
        accountEndpoint: appSettings.ProviderAccountEndpoint,
        accountKey: appSettings.ProviderAccountKey,
        databaseName: appSettings.DatabaseName)
);

builder.Services.AddTransient<MetadataRepository>();

// Add built-in middleware
builder.Services.AddControllers().ConfigureApiBehaviorOptions(x => { x.SuppressMapClientErrors = true; });
builder.Services.AddEndpointsApiExplorer();

var app = builder.Build();

// Add Swagger to the HTTP request pipeline
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    var clientId = !string.IsNullOrWhiteSpace(appSettings.AzureAd.ClientId) ? 
        appSettings.AzureAd.ClientId : Guid.NewGuid().ToString();

    c.OAuthClientId(clientId);
    c.OAuthUsePkce();
    c.OAuthScopeSeparator(" ");
    c.DefaultModelsExpandDepth(-1);
});

app.UseHttpsRedirection();

// Add the CORS policy to the HTTP Request pipeline
app.UseCors("AllowAnyOrigin");

// Add authentication and authorization to the HTTP Request pipeline
app.UseAuthentication();
app.UseAuthorization();

// Add the API endpoint controllers to the HTTP Request pipeline
app.MapControllers();

app.Run();
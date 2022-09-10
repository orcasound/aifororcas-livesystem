using AIForOrcas.Server.BL.Context;
using AIForOrcas.Server.BL.Services;
using AIForOrcas.Server.Extensions;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi.Models;
using System;
using System.IO;
using System.Reflection;

var builder = WebApplication.CreateBuilder(args);

// Add authentication/authorization middleware (AuthMiddleWareExtensions)
builder.ConfigureCors();

builder.Services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseCosmos(
        accountEndpoint: builder.Configuration["AccountEndpoint"],
        accountKey: builder.Configuration["AccountKey"],
        databaseName: builder.Configuration["DatabaseName"])
);

builder.Services.AddTransient<MetadataRepository>();

builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v2", new OpenApiInfo
    {
        Title = "AI For Orcas API",
        Version = "v2",
        Description = "REST API for interacting with the AI For Orcas CosmosDB."
    });

    // Set the comments path for the controllers.
    var baseXmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    var baseXmlPath = Path.Combine(AppContext.BaseDirectory, baseXmlFile);
    c.IncludeXmlComments(baseXmlPath);

    // Set the comments path for the schemas.
    var schemaXmlPath = Path.Combine(AppContext.BaseDirectory, "AIForOrcas.DTO.xml");
    c.IncludeXmlComments(schemaXmlPath);
});

builder.Services.AddCors(o =>
{
    o.AddPolicy("CorsPolicy", policy => policy
    .WithOrigins(builder.Configuration["AllowedOrigin"])
    .AllowAnyMethod()
    .AllowAnyHeader());
});

builder.Services.AddControllers();

var app = builder.Build();

// This makes the contents of wwwroot available to be served
app.UseStaticFiles();

app.UseHttpsRedirection();

app.UseRouting();

app.UseAuthorization();

app.UseSwagger();

app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v2/swagger.json", "AI For Orcas API");
    c.RoutePrefix = string.Empty;
});

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers();
});

app.Run();
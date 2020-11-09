using AIForOrcas.Server.BL.Context;
using AIForOrcas.Server.BL.Services;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi.Models;
using System;
using System.IO;
using System.Reflection;

namespace AIForOrcas.Server
{
	public class Startup
	{
		public Startup(IConfiguration configuration)
		{
			Configuration = configuration;
		}

		public IConfiguration Configuration { get; }

		public void ConfigureServices(IServiceCollection services)
		{
			services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();

			services.AddDbContext<ApplicationDbContext>(options =>
				options.UseCosmos(
					accountEndpoint: Configuration["AccountEndpoint"],
					accountKey: Configuration["AccountKey"],
					databaseName: Configuration["DatabaseName"])
			);

			services.AddTransient<MetadataRepository>();

			services.AddSwaggerGen(c =>
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
			
			services.AddCors(o =>
			{
				o.AddPolicy("CorsPolicy", builder => builder
				.WithOrigins(Configuration["AllowedOrigin"])
				.AllowAnyMethod()
				.AllowAnyHeader());
			});


			services.AddControllers();
		}

		public void Configure(IApplicationBuilder app)
		{
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

			//app.UseCors(builder =>
			//{
			//	builder.WithOrigins(Configuration["allowedOrigin"]);
			//});
		}
	}
}

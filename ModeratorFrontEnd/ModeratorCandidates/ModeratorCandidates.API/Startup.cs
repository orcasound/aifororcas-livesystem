using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.OpenApi.Models;
using ModeratorCandidates.API.Services;

namespace ModeratorCandidates.API
{
	public class Startup
	{
		public Startup(IConfiguration configuration)
		{
			Configuration = configuration;
		}

		public IConfiguration Configuration { get; }

		// This method gets called by the runtime. Use this method to add services to the container.
		public void ConfigureServices(IServiceCollection services)
		{
			services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();

			services.AddDbContext<ApplicationDbContext>(options =>
				options.UseCosmos(
					accountEndpoint: Configuration["accountEndpoint"],
					accountKey: Configuration["accountKey"],
					databaseName: Configuration["databaseName"])
			);

			// This calls the cosmos db
			services.AddTransient<MetadataRepository>();

			services.AddSwaggerGen(c =>
			{
				c.SwaggerDoc("v1", new OpenApiInfo { Title = "Moderator Candidates API", Version = "v1" });
			});

			// TODO: Will have to be tweeked down once security is addressed
			services.AddCors(o =>
			{
				o.AddPolicy("CorsPolicy", builder => builder.AllowAnyOrigin()
				.AllowAnyMethod()
				.AllowAnyHeader());
			});

			services.AddControllers();
		}

		// This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
		public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
		{
			if (env.IsDevelopment())
			{
				app.UseDeveloperExceptionPage();
			}

			// This makes the contents of wwwroot available to be served
			app.UseStaticFiles();

			app.UseHttpsRedirection();

			app.UseRouting();

			app.UseAuthorization();

			app.UseSwagger();

			app.UseSwaggerUI(c =>
			{
				c.SwaggerEndpoint("/swagger/v1/swagger.json", "Moderator Candidates API");
			});

			app.UseEndpoints(endpoints =>
			{
				endpoints.MapControllers();
			});
		}
	}
}


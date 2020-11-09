using AIForOrcas.Client.BL.Helpers;
using AIForOrcas.Client.BL.Services;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.AzureAD.UI;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace AIForOrcas.Client.Web
{
	public class Startup
	{
		public Startup(IConfiguration configuration)
		{
			Configuration = configuration;
		}

		public IConfiguration Configuration { get; }

		// This method gets called by the runtime. Use this method to add services to the container.
		// For more information on how to configure your application, visit https://go.microsoft.com/fwlink/?LinkID=398940
		public void ConfigureServices(IServiceCollection services)
		{
			// Following 3 services enable authentication/authorization through Azure AD

			services.AddAuthentication(AzureADDefaults.AuthenticationScheme)
				.AddAzureAD(options => Configuration.Bind("AzureAd", options));

			services.AddControllersWithViews();

			services.AddAuthorization(options =>
			{
				options.AddPolicy("ModeratorRole", policyBuilder =>
					policyBuilder.RequireClaim("groups", Configuration["AzureADGroup:ModeratorGroupId"]));
			});

			services.AddRazorPages();
			services.AddServerSideBlazor();

			services.AddHttpContextAccessor();

			services.AddHttpClient<IDetectionService, DetectionService>(client =>
			{
				client.BaseAddress = new System.Uri(Configuration["APIUrl"]);
			});

			services.AddHttpClient<IMetricsService, MetricsService>(client =>
			{
				client.BaseAddress = new System.Uri(Configuration["APIUrl"]);
			});

			services.AddScoped<IdentityHelper>();
		}

		// This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
		public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
		{
			if (env.IsDevelopment())
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
		}
	}
}

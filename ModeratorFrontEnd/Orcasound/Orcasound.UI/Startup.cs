using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.AzureAD.UI;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Orcasound.UI.Services;
using System;

namespace Orcasound.UI
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
			services.AddAuthentication(AzureADDefaults.AuthenticationScheme)
				.AddAzureAD(options => Configuration.Bind("AzureAd", options));

			services.AddControllersWithViews();

			services.AddAuthorization(options =>
			{
				options.AddPolicy("ModeratorGroupRole", policyBuilder =>
					policyBuilder.RequireClaim("groups", "bb64ad99-f25d-4f08-a61b-38851ac3d263"));
			});

			//services.AddControllersWithViews(options =>
			//{
			//	var policy = new AuthorizationPolicyBuilder()
			//		.RequireClaim("groups",
			//			Configuration.GetValue<string>("AzureADGroup:ModeratorGroupId"))
			//		.Build();
			//	options.Filters.Add(new AuthorizeFilter(policy));
			//});

			services.AddRazorPages();
			services.AddServerSideBlazor();

			services.AddHttpClient<ICandidateService, APICandidateService>(client =>
			{
				client.BaseAddress = new Uri("https://localhost:44319/");

				// client.BaseAddress = new Uri("https://moderatorcandidates.azurewebsites.net/");
			});

			services.AddHttpContextAccessor();

			// Optional for debugging
			services.AddServerSideBlazor(o => o.DetailedErrors = true);
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
				endpoints.MapControllers();
				endpoints.MapBlazorHub();
				endpoints.MapFallbackToPage("/_Host");
			});
		}
	}
}

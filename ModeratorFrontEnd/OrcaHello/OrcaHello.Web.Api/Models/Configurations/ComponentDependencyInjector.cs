using OrcaHello.Web.Api.Brokers.Hydrophones;

namespace OrcaHello.Web.Api.Models.Configurations
{
    /// <summary>
    /// Static class for performing dependency injection of the various components
    /// of the environment.
    /// </summary>
    [ExcludeFromCodeCoverage]
    public static class ComponentDependencyInjector
    {
        public static void AddBrokers(WebApplicationBuilder builder)
        {
            builder.Services.AddSingleton<IStorageBroker, StorageBroker>();
            builder.Services.AddTransient<ILoggingBroker, LoggingBroker>();
            builder.Services.AddSingleton<IHydrophoneBroker, HydrophoneBroker>();
        }

        public static void AddFoundationServices(WebApplicationBuilder builder)
        {
            builder.Services.AddTransient<IMetadataService, MetadataService>();
            builder.Services.AddTransient<IHydrophoneService, HydrophoneService>();
        }

        public static void AddOrchestrationServices(WebApplicationBuilder builder)
        {
            builder.Services.AddTransient<ICommentOrchestrationService, CommentOrchestrationService>();
            builder.Services.AddTransient<IDetectionOrchestrationService, DetectionOrchestrationService>();
            builder.Services.AddTransient<IInterestLabelOrchestrationService, InterestLabelOrchestrationService>();
            builder.Services.AddTransient<IHydrophoneOrchestrationService, HydrophoneOrchestrationService>();
            builder.Services.AddTransient<IMetricsOrchestrationService, MetricsOrchestrationService>();
            builder.Services.AddTransient<IModeratorOrchestrationService, ModeratorOrchestrationService>();
            builder.Services.AddTransient<ITagOrchestrationService, TagOrchestrationService>();
        }

        // Allow CORS access from anywhere
        public static void ConfigureCors(WebApplicationBuilder builder)
        {
            builder.Services.AddCors(o => o.AddPolicy("AllowAnyOrigin",
                builder =>
                {
                    builder.AllowAnyOrigin()
                            .AllowAnyMethod()
                            .AllowAnyHeader();
                }));
        }

        // Implement Jwt Authentication
        public static void ConfigureJwtAuthentication(WebApplicationBuilder builder)
        {
            builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
                .AddMicrosoftIdentityWebApi(builder.Configuration.GetSection("AppSettings:AzureAd"));

            builder.Services.Configure<JwtBearerOptions>(
                JwtBearerDefaults.AuthenticationScheme, options =>
                {
                    options.TokenValidationParameters.NameClaimType = "name";
                });
        }

        // Implement Moderator Authorization policy
        public static void ConfigureModeratorPolicy(WebApplicationBuilder builder, AppSettings appSettings)
        {
            builder.Services.AddAuthorization(options =>
            {
                var moderatorGroupId = !string.IsNullOrWhiteSpace(appSettings.AzureAd.ModeratorGroupId)
                    ? appSettings.AzureAd.ModeratorGroupId : Guid.NewGuid().ToString();

                options.AddPolicy("Moderators",
                    policy => policy.RequireClaim("groups", moderatorGroupId));
            });
        }

        // Set up Swagger so users can use OAuth to authenticate against it
        public static void ConfigureSwagger(WebApplicationBuilder builder, AppSettings appSettings)
        {
            var instance = !string.IsNullOrWhiteSpace(appSettings.AzureAd.Instance) ?
                appSettings.AzureAd.Instance : string.Empty;
            var tenantId = !string.IsNullOrWhiteSpace(appSettings.AzureAd.TenantId) ?
                appSettings.AzureAd.TenantId : Guid.NewGuid().ToString();
            var clientId = !string.IsNullOrWhiteSpace(appSettings.AzureAd.ClientId) ?
                appSettings.AzureAd.ClientId : Guid.NewGuid().ToString();
            var scopes = !string.IsNullOrWhiteSpace(appSettings.AzureAd.Scopes) ?
                appSettings.AzureAd.Scopes : string.Empty;

            var authUrl = $"{instance}/{tenantId}/oauth2/v2.0/authorize";
            var tokenUrl = $"{instance}/{tenantId}/oauth2/v2.0/token";

            builder.Services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc("v1", new OpenApiInfo
                {
                    Title = "AI For Orcas API",
                    Version = "v1.2",
                    Description = "REST API for accessing and updating AI For Orcas detections, tags, and metrics."
                });

                c.EnableAnnotations();

                // Set the comments path for the controllers.
                var baseXmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
                var baseXmlPath = Path.Combine(AppContext.BaseDirectory, baseXmlFile);
                c.IncludeXmlComments(baseXmlPath, includeControllerXmlComments: true);

                c.AddSecurityDefinition("OAuth2", new OpenApiSecurityScheme
                {
                    Description = "OAuth2.0 Auth with Proof Key for Code Exchange (PKCE)",
                    Name = "OAuth2",
                    Type = SecuritySchemeType.OAuth2,
                    Flows = new OpenApiOAuthFlows
                    {
                        AuthorizationCode = new OpenApiOAuthFlow
                        {
                            AuthorizationUrl = new Uri(authUrl),
                            TokenUrl = new Uri(tokenUrl),
                            Scopes = new Dictionary<string, string>
                        {
                            { $"api://{clientId}/{scopes}", "For accessing endpoints requiring authorization." }
                        }
                        }
                    }
                });

                c.AddSecurityRequirement(new OpenApiSecurityRequirement
            {
                {
                    new OpenApiSecurityScheme
                    {
                        Reference = new OpenApiReference { Type = ReferenceType.SecurityScheme, Id = "OAuth2" }
                    },
                    new[] { scopes }
                }
            });
            });
        }
    }
}
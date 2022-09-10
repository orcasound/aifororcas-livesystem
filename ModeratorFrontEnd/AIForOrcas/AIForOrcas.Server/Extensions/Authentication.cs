using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;

namespace AIForOrcas.Server.Extensions;

public static class Authentication
{
    // Allow CORS access from anywhere
    public static void ConfigureCors(this WebApplicationBuilder builder)
    {
        builder.Services.AddCors(o => o.AddPolicy("AllowAnyOrigin",
            builder =>
            {
                builder.AllowAnyOrigin()
                        .AllowAnyMethod()
                        .AllowAnyHeader();
            }));
    }
}

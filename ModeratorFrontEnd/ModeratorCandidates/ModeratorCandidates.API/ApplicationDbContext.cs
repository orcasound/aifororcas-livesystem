using Microsoft.EntityFrameworkCore;
using ModeratorCandidates.Shared.Models;

namespace ModeratorCandidates.API
{
    public class ApplicationDbContext : DbContext
    {
		public DbSet<Metadata> Metadata { get; set; }

		public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
			: base(options)
		{ }

		protected override void OnModelCreating(ModelBuilder modelBuilder)
		{
			modelBuilder.Entity<Metadata>().ToContainer("metadata");
			modelBuilder.Entity<Metadata>().OwnsOne(p => p.location);
			modelBuilder.Entity<Metadata>().OwnsMany(p => p.predictions);
			modelBuilder.Entity<Metadata>().HasNoDiscriminator();
			base.OnModelCreating(modelBuilder);
		}
	}
}

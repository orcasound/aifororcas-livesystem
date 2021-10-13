using AIForOrcas.Server.BL.Models.CosmosDB;
using System;
using System.Linq;

namespace AIForOrcas.Server.Helpers
{
    public static class MetadataFilters
	{
		public static int DefaultRecordsPerPage = 5;

		public static void ApplyTimeframeFilter(ref IQueryable<Metadata> queryable, string timeframe)
		{
			if (!string.IsNullOrWhiteSpace(timeframe))
			{
				timeframe = timeframe.ToLower();

				if (timeframe != "all")
				{
					var now = DateTime.Now;

					if (timeframe == "30m")
						now = now.AddMinutes(-30);

					if (timeframe == "3h")
						now = now.AddHours(-3);

					if (timeframe == "6h")
						now = now.AddHours(-6);

					if (timeframe == "24h")
						now = now.AddHours(-24);

					if (timeframe == "1w")
						now = now.AddDays(-7);

					if (timeframe == "1m")
						now = now.AddDays(-30);

					queryable = queryable.Where(x => x.timestamp >= now);

				}
			}
		}

		public static void ApplyModeratorFilter(ref IQueryable<Metadata> queryable, string moderator)
		{
			if (!string.IsNullOrWhiteSpace(moderator))
			{
				queryable = queryable.Where(x => x.moderator == moderator);
			}
		}

		public static void ApplyLocationFilter(ref IQueryable<Metadata> queryable, string location)
		{
			if(!string.IsNullOrWhiteSpace(location))
			{
				queryable = queryable.Where(x => x.location.name == location);
			}
		}

		public static void ApplyReviewedFilter(ref IQueryable<Metadata> queryable, bool reviewed)
		{
			queryable = queryable.Where(x => x.reviewed == reviewed);
		}

		public static void ApplyFoundFilter(ref IQueryable<Metadata> queryable, string foundState)
		{
			queryable = queryable.Where(x => x.SRKWFound == foundState);
		}
	}
}

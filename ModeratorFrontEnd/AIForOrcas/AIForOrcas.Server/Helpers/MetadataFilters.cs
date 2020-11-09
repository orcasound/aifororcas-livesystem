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

					if (timeframe == "24h")
						now = now.AddHours(-24);

					if (timeframe == "1w")
						now = now.AddDays(-7);

					if (timeframe == "1m")
						now = now.AddDays(-30);

					queryable = queryable.Where(x => DateTime.Parse(x.timestamp) >= now);
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

		public static void ApplyReviewedFilter(ref IQueryable<Metadata> queryable, bool reviewed)
		{
			queryable = queryable.Where(x => x.reviewed == reviewed);
		}

		public static void ApplyFoundFilter(ref IQueryable<Metadata> queryable, string foundState)
		{
			queryable = queryable.Where(x => x.SRKWFound.ToLower() == foundState.ToLower());
		}

		public static void ApplyConfidenceSortFilter(ref IQueryable<Metadata> queryable, string sortOrder)
		{
			if (sortOrder == "asc")
				queryable = queryable.OrderBy(x => x.whaleFoundConfidence)
					.ThenBy(x => x.timestamp)
					.ThenBy(x => x.id);

			if (sortOrder == "desc")
				queryable = queryable.OrderByDescending(x => x.whaleFoundConfidence)
					.ThenBy(x => x.timestamp)
					.ThenBy(x => x.id);
		}

		public static void ApplyTimestampSortFilter(ref IQueryable<Metadata> queryable, string sortOrder)
		{
			if (sortOrder == "asc")
				queryable = queryable.OrderBy(x => x.timestamp)
					.ThenByDescending(x => x.whaleFoundConfidence)
					.ThenBy(x => x.id);

			if (sortOrder == "desc")
				queryable = queryable.OrderByDescending(x => x.timestamp)
					.ThenByDescending(x => x.whaleFoundConfidence)
					.ThenBy(x => x.id);
		}

		public static void ApplyPaginationFilter(ref IQueryable<Metadata> queryable, int page, int take)
		{
			var skip = page > 0 ? page - 1 : 0;
			var recordsPerPage = take > 0 ? take : DefaultRecordsPerPage;

			queryable = queryable
				.Skip(skip * recordsPerPage)
				.Take(recordsPerPage);
		}
	}
}

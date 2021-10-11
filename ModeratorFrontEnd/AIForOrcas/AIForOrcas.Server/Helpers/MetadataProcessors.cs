using AIForOrcas.DTO.API;
using AIForOrcas.Server.BL.Models.CosmosDB;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AIForOrcas.Server.Helpers
{
	public static class MetadataProcessors
	{
		public static (int ReviewedCount, int UnreviewedCount) GetReviewed(IQueryable<Metadata> queryable)
		{
			var status = queryable.GroupBy(n => n.reviewed);

			var reviewed = status.Where(x => x.Key == true)
				.Select(x => x.Count()).FirstOrDefault();

			var unreviewed = status.Where(x => x.Key == false)
				.Select(x => x.Count()).FirstOrDefault();

			return (reviewed, unreviewed);
		}

		public static (int ConfirmedCount, int FalseCount, int UnknownCount) GetResults(IQueryable<Metadata> queryable)
		{
			// grab results metrics
			var results = queryable.GroupBy(n => n.SRKWFound);

			var confirmed = results.Where(x => x.Key == "yes")
				.Select(x => x.Count()).FirstOrDefault();

			var unconfirmed = results.Where(x => x.Key == "no")
				.Select(x => x.Count()).FirstOrDefault();

			var unknown = results.Where(x => x.Key == "don't know")
				.Select(x => x.Count()).FirstOrDefault();

			return (confirmed, unconfirmed, unknown);
		}

		public static List<MetricsComment> GetComments(IQueryable<Metadata> queryable, string status)
		{
			var results = new List<MetricsComment>();

			queryable
				.Where(x => x.SRKWFound == status && !string.IsNullOrWhiteSpace(x.comments))
				.ToList().ForEach(x =>
				{
					results.Add(new MetricsComment()
					{
						Comment = x.comments,
						Moderator = x.moderator,
						Timestamp = DateTime.Parse(x.dateModerated),
						Id = x.id
					});
				});

			return results.OrderByDescending(x => x.Timestamp).ToList();
		}

		public static List<MetricsTag> GetTags(IQueryable<Metadata> queryable)
		{
			var results = new List<MetricsTag>();

			queryable
				.Where(x => !string.IsNullOrWhiteSpace(x.tags))
				.ToList().ForEach(y =>
				{
					var id = y.id;
					y.tags.Split(";")
					.ToList().ForEach(z =>
					{
						var tag = results.Where(t => t.Tag == z.ToUpper()).FirstOrDefault();
						if (tag != null)
						{
							tag.Ids.Add(id);
						}
						else
						{
							var newTag = new MetricsTag()
							{
								Tag = z.ToUpper()
							};
							newTag.Ids.Add(id);
							results.Add(newTag);
						}
					});
				});

			return results.OrderBy(x => x.Tag).ToList();
		}

		public static Detection ToDetection(Metadata metadata)
		{
			var detection = new Detection()
			{
				Id = string.IsNullOrEmpty(metadata.id) ? Guid.NewGuid().ToString() : metadata.id,
				SpectrogramUri = string.IsNullOrWhiteSpace(metadata.imageUri) ? string.Empty : metadata.imageUri,
				AudioUri = string.IsNullOrWhiteSpace(metadata.audioUri) ? string.Empty : metadata.audioUri,
				Reviewed = metadata.reviewed,
				Confidence = metadata.whaleFoundConfidence,
				Found = string.IsNullOrWhiteSpace(metadata.SRKWFound) ? "No" : metadata.SRKWFound,
				Timestamp = string.IsNullOrWhiteSpace(metadata.timestamp) ? DateTime.Now : DateTime.Parse(metadata.timestamp),
				Comments = metadata.comments,
				Tags = metadata.tags,
				Moderated = string.IsNullOrWhiteSpace(metadata.dateModerated) ? DateTime.Now : DateTime.Parse(metadata.dateModerated),
				Moderator = metadata.moderator,
				Location = new DTO.API.Location()
				{
					Name = metadata.location.name,
					Longitude = metadata.location.longitude,
					Latitude = metadata.location.latitude
				}
			};

			if (metadata.predictions?.Count > 0)
			{
				metadata.predictions.ForEach(x =>
				{
					detection.Annotations.Add(new Annotation()
					{
						Id = x.id,
						Confidence = x.confidence,
						StartTime = x.startTime,
						EndTime = x.startTime + x.duration,
					});
				});
			}

			return detection;
		}
	}
}

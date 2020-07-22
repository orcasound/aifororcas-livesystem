using System;

namespace ModeratorCandidates.Shared.Models
{
	public class AIClipMetadataReviewResult
	{
		public string Id { get; set; }
		public string comments { get; set; }
		public string status { get; set; }
		public string tags { get; set; }
		public string moderator { get; set; }
		public DateTime dateModerated { get; set; }
	}
}

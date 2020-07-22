using System.Collections.Generic;

namespace ModeratorCandidates.Shared.Models
{
	// Raw format for results stored as .json files
	public class JsonClipMetadata
	{
		public string uri { get; set; }
		public string absolute_time { get; set; }
		public string source_guid { get; set; }
		public List<JsonAnnotation> annotations { get; set; }
	}
}

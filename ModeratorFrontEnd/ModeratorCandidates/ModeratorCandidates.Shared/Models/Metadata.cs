using System;
using System.Collections.Generic;
using System.Text;

namespace ModeratorCandidates.Shared.Models
{
    public class Metadata
    {
		public string id { get; set; }
		public string audioUri { get; set; }
		public string imageUri { get; set; }
		public bool reviewed { get; set; }
		public string timestamp { get; set; }
		public decimal whaleFoundConfidence { get; set; }
		public Location location { get; set; }
		public List<Prediction> predictions { get; set; }
		public string SRKWFound { get; set; }
		public string comments { get; set; }
		public string dateModerated { get; set; }
		public string moderator { get; set; }
		public string tags { get; set; }
	}
}

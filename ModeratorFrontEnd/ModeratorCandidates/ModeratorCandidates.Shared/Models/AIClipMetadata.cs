using System;
using System.Collections.Generic;

namespace ModeratorCandidates.Shared.Models
{
	public class AIClipMetadata
	{
		public string id { get; set; }
		public string audioUri { get; set; }
		public string imageUri { get; set; }
		public AILocation location { get; set; }
		public DateTime timestamp { get; set; }
		public List<AIAnnotation> annotations { get; set; }
		public string status { get; set; }
		public string found { get; set; }
		public string comments { get; set; }
		public decimal averageConfidence { get; set; }
		public string moderator { get; set; }
		public DateTime dateModerated { get; set; }
		public string tags { get; set; }
	}
}

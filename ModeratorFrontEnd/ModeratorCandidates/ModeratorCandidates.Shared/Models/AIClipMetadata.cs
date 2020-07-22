using System;
using System.Collections.Generic;
using System.Linq;

namespace ModeratorCandidates.Shared.Models
{
	// TODO: Per Akash, this may need to be changed:
	// class name > EventSegment
	// audioUri > AudioSegment
	// imageUrl > SpectrogranSegment
	// annotations > EventMetadata?
	// status > EventStatus

	// Not sure if it is a hard requirement or a suggestion that needs to be discussed

	public class AIClipMetadata
	{
		public string id { get; set; }
		public string audioUri { get; set; }
		public string imageUri { get; set; }
		public AILocation location { get; set; }
		public DateTime timestamp { get; set; }
		public List<AIAnnotation> annotations { get; set; }
		public string status { get; set; }
		public string comments { get; set; }
		public decimal averageConfidence {
			get
			{
				var total = annotations.Select(a => a.confidence).Sum();
				return total / annotations.Count;
			}
		}

	}
}

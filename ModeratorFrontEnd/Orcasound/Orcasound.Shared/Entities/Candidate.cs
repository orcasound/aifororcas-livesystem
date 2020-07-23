using System;
using System.Collections.Generic;
using System.Linq;

namespace Orcasound.Shared.Entities
{
	public class Candidate
	{
		public string Id { get; set; }
		public string AudioUri { get; set; }
		public string ImageUri { get; set; }
		public DateTime Timestamp { get; set; }
		public string Status { get; set; }
		public string Found { get; set; }
		public string Comments { get; set; }
		public string Tags { get; set; }
		public string Moderator { get; set;}
		public DateTime DateModerated { get; set; }
		public Location Location { get; set; }

		public List<Annotation> Annotations { get; set; }

		public decimal AverageConfidence
		{
			get
			{
				var total = Annotations.Select(a => a.Confidence).Sum();
				return (total / Annotations.Count) * 100;
			}
		}

		public int Detections
		{
			get
			{
				return Annotations.Count;
			}
		}
	}
}

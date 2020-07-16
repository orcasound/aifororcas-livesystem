using System;
using System.Collections.Generic;

namespace Orcasound.Shared.Entities
{
	public class Candidate
	{
		public int Id { get; set; }
		public DateTime Timestamp { get; set; }
		public string Source { get; set; }
		public string Node { get; set; }
		public double Longitute { get; set; }
		public double Latitude { get; set; }
		public double Probability { get; set; }
		public string Description { get; set; }
		public string ApprovalComments { get; set; }
		public string RejectionComments { get; set; }
		public List<string> Tags { get; set; } = new List<string>();

		public string TagString
		{
			get
			{
				return string.Join(", ", Tags);
			}
		}
	}
}

using System.Collections.Generic;

namespace AIForOrcas.DTO.API
{
	/// <summary>
	/// Information about entered tag.
	/// </summary>
	public class MetricsTag
	{
		/// <summary>
		/// Tag name.
		/// </summary>
		public string Tag { get; set; }

		/// <summary>
		/// List of detection unique Ids associated with this tag.
		/// </summary>
		public List<string> Ids { get; set; } = new List<string>();
	}
}

using System;

namespace AIForOrcas.DTO.API
{
	/// <summary>
	/// Detection data to be updated.
	/// </summary>
	public class DetectionUpdate
	{
		/// <summary>
		/// The detection's unique ID.
		/// </summary>
		public string Id { get; set; }

		/// <summary>
		/// Comments provided by the human moderator.
		/// </summary>
		public string Comments { get; set; }

		/// <summary>
		/// Tags provided by the human moderator.
		/// </summary>
		public string Tags { get; set; }

		/// <summary>
		/// Identity of the human moderator (User Principal Name for AzureAD) reviewing the detection.
		/// </summary>
		public string Moderator { get; set; }

		/// <summary>
		/// Date and time when the detection was moderated.
		/// </summary>
		public DateTime Moderated { get; set; }

		/// <summary>
		/// Flag indicating whether or not the detection has been reviewed.
		/// </summary>
		public bool Reviewed { get; set; }

		/// <summary>
		/// Indicates whether whale sounds were heard in the detection.
		/// </summary>
		/// <example>yes, no, don't know</example>
		public string Found { get; set; }
	}
}

using System;

namespace AIForOrcas.DTO.API
{
	/// <summary>
	/// Information about entered comment.
	/// </summary>
	public class MetricsComment
	{
		/// <summary>
		/// The text of the comment.
		/// </summary>
		public string Comment { get; set; }

		/// <summary>
		/// The detection's unique Id.
		/// </summary>
		public string Id { get; set; }

		/// <summary>
		/// Date and time when the comment was submitted.
		/// </summary>
		public DateTime Timestamp { get; set; }

		/// <summary>
		/// Identity of the human moderator (User Principal Name for AzureAD) submitting the comment.
		/// </summary>
		public string Moderator { get; set; }
	}
}

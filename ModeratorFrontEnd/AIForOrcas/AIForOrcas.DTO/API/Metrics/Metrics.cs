using System.Collections.Generic;

namespace AIForOrcas.DTO.API
{
	/// <summary>
	/// Activity metrics for the entire system.
	/// </summary>
	public class Metrics
	{
		/// <summary>
		/// Activity timeframe (30m, 24h, etc.).
		/// </summary>
		public string Timeframe { get; set; }

		/// <summary>
		/// Number of reviewed detections in timeframe.
		/// </summary>
		public int Reviewed { get; set; }

		/// <summary>
		/// Number of detections not reviewed in timeframe.
		/// </summary>
		public int Unreviewed { get; set; }

		/// <summary>
		/// Number of detections in timeframe confirmed by human moderator to have whale sound.
		/// </summary>
		public int ConfirmedDetection { get; set; }

		/// <summary>
		/// Number of detections in timeframe confirmed by human moderator to not have whale sound.
		/// </summary>
		public int FalseDetection { get; set; }

		/// <summary>
		/// Number of detections in timeframe where human moderator could not determine if there was whale sound.
		/// </summary>
		public int UnknownDetection { get; set; }

		/// <summary>
		/// List of all comments in timeframe concerning confirmed detections.
		/// </summary>
		public List<MetricsComment> ConfirmedComments { get; set; } = new List<MetricsComment>();

		/// <summary>
		/// List of all comments in timeframe concerning unconfirmed or unknown detections.
		/// </summary>
		public List<MetricsComment> UnconfirmedComments { get; set; } = new List<MetricsComment>();

		/// <summary>
		/// List of all tags in timeframe.
		/// </summary>
		public List<MetricsTag> Tags { get; set; } = new List<MetricsTag>();

		/// <summary>
		/// Formatted detections reviewed/unreviewed for passing to JSInterop.
		/// </summary>
		public string DetectionsArray => $"[{Reviewed}, {Unreviewed}]";

		/// <summary>
		/// Formatted detection results for passing to JSInterop.
		/// </summary>
		public string DetectionResultsArray => $"[{ConfirmedDetection}, {FalseDetection}, {UnknownDetection}]";


		/// <summary>
		/// Flag to be set if no metrics retrieved.
		/// </summary>
		public bool HasContent { get; set; } = false;
	}
}

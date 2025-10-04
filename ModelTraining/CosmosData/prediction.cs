public class Prediction
{
	public string id { get; set; }
	public string modelId { get; set; }
	public string audioUri { get; set; }
	public string imageUri { get; set; }
	public bool reviewed { get; set; }
	public string timestamp { get; set; }
	public double whaleFoundConfidence { get; set; }
	public string location_id { get; set; }
	public string location_lat { get; set; }
	public string location_name { get; set; }
	public string location_long { get; set; }
	public string source_guid { get; set; }
	public string prediction_id { get; set; }
	public string prediction_startTime { get; set; }
	public string prediction_duration { get; set; }
	public string prediction_confidence { get; set; }
}

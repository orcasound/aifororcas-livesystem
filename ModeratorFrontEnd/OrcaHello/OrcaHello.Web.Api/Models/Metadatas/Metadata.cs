using Newtonsoft.Json;

namespace OrcaHello.Web.Api.Models
{
    /// <summary>
    /// Metadata collected from a hydrophone sampling that might contain whale sounds.
    /// </summary>
    [ExcludeFromCodeCoverage]
    public class Metadata
    {
        /// <summary>
        /// The metadata's generated unique Id.
        /// </summary>
        /// <example>00000000-0000-0000-0000-000000000000</example>
        [JsonProperty("id", NullValueHandling = NullValueHandling.Ignore)]
        public string Id { get; set; }

        /// <summary>
        /// The metadata's moderation state (Unreviewed, Positive, Negative, Unknown).
        /// </summary>
        /// <example>Positive</example>
        [JsonProperty("state", NullValueHandling = NullValueHandling.Ignore)]
        public string State { get; set; }

        /// <summary>
        /// The name of the hydrophone where the metadata was collected.
        /// </summary>
        /// <example>Haro Strait</example>
        [JsonProperty("locationName", NullValueHandling = NullValueHandling.Ignore)]
        public string LocationName { get; set; }

        /// <summary>
        /// URI of the metadata's audio file (.wav) in blob storage.
        /// </summary>
        /// <example>https://storagesite.blob.core.windows.net/audiowavs/audiofilename.wav</example>
        [JsonProperty("audioUri", NullValueHandling = NullValueHandling.Ignore)]
        public string AudioUri { get; set; }

        /// <summary>
        /// URI of the metadata's image file (.png) in blob storage.
        /// </summary>
        /// <example>https://storagesite.blob.core.windows.net/spectrogramspng/imagefilename.png</example>
        [JsonProperty("imageUri", NullValueHandling = NullValueHandling.Ignore)]
        public string ImageUri { get; set; }

        /// <summary>
        /// Date and time of when the detection occurred.
        /// </summary>
        /// <example>2020-09-30T11:03:56.057346Z</example>
        [JsonProperty("timestamp")]
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Calculated average confidence that the metadata contains a whale sound.
        /// </summary>
        /// <example>84.39</example>
        [JsonProperty("whaleFoundConfidence")]
        public decimal WhaleFoundConfidence { get; set; }

        /// <summary>
        /// Detailed location of the hydrophone that collected the metadata.
        /// </summary>
        [JsonProperty("location")]
        public Location Location { get; set; }

        /// <summary>
        /// List of sections within the collected audio that might contain whale sounds.
        /// </summary>
        [JsonProperty("predictions")]
        public List<Prediction> Predictions { get; set; } = new List<Prediction>();

        /// <summary>
        /// Any text comments entered by the human moderator during review.
        /// </summary>
        /// <example>Clear whale sounds detected.</example>
        [JsonProperty("comments", NullValueHandling = NullValueHandling.Ignore)]
        public string Comments { get; set; }

        /// <summary>
        /// Date and time of when the metadata was reviewed by the human moderator.
        /// </summary>
        /// <example>2020-09-30T11:03:56Z</example>
        [JsonProperty("dateModerated", NullValueHandling = NullValueHandling.Ignore)]
        public string DateModerated { get; set; }

        /// <summary>
        /// Identity of the human moderator (User Principal Name for AzureAD) performing the review.
        /// </summary>
        /// <example>user@gmail.com</example>
        [JsonProperty("moderator", NullValueHandling = NullValueHandling.Ignore)]
        public string Moderator { get; set; }

        /// <summary>
        /// Any descriptive tags entered by the moderator during the review.
        /// </summary>
        /// <example>S7 and S10</example>
        [JsonProperty("tags")]
        public List<string> Tags { get; set; } = new List<string>();

        /// <summary>
        /// A flag indicating that the metadata might be a special item of interest for the public facing pages.
        /// </summary>
        /// <example>other whales</example>
        [JsonProperty("interestLabel", NullValueHandling = NullValueHandling.Ignore)]
        public string InterestLabel { get; set; }
    }

    /// <summary>
    /// Geographical location of the hydrophone that collected the metadata.
    /// </summary>
    [ExcludeFromCodeCoverage]
    public class Location
    {
        /// <summary>
        /// The id of the hydrophone location.
        /// </summary>
        /// <example>hydrophone1</example>
        [JsonProperty("id", NullValueHandling = NullValueHandling.Ignore)]
        public string Id { get; set; }

        /// <summary>
        /// Name of the hydrophone location.
        /// </summary>
        /// <example>Haro Strait</example>
        [JsonProperty("name", NullValueHandling = NullValueHandling.Ignore)]
        public string Name { get; set; }

        /// <summary>
        /// Longitude of the hydrophone's location.
        /// </summary>
        /// <example>-123.2166658</example>
        [JsonProperty("longitude")]
        public double Longitude { get; set; }

        /// <summary>
        /// Latitude of the hydrophone's location.
        /// </summary>
        /// <example>48.5499978</example>
        [JsonProperty("latitude")]
        public double Latitude { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class Prediction
    {
        /// <summary>
        /// Unique identifier (within the audio file) of the annotation.
        /// </summary>
        /// <example>1</example>
        [JsonProperty("id")]
        public int Id { get; set; }

        /// <summary>
        /// Start time (within the audio file) of the annotation as measured in seconds.
        /// </summary>
        /// <example>35</example>
        [JsonProperty("startTime")]
        public decimal StartTime { get; set; }

        /// <summary>
        /// Duration (within the audio file) of the annotation as measured in seconds.
        /// </summary>
        /// <example>37.5</example>
        [JsonProperty("duration")]
        public decimal Duration { get; set; }

        /// <summary>
        /// Calculated confidence that the annotation contains a whale sound.
        /// </summary>
        /// <example>84.39</example>
        [JsonProperty("confidence")]
        public decimal Confidence { get; set; }
    }
}

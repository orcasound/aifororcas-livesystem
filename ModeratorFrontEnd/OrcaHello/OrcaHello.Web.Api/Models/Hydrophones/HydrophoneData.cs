namespace OrcaHello.Web.Api.Models
{
    /// <summary>
    /// Hydrophone data retrieved from the provider service.
    /// </summary>
    [ExcludeFromCodeCoverage]
    public class HydrophoneRootObject
    {
        [JsonProperty("data", NullValueHandling = NullValueHandling.Ignore)]
        public List<HydrophoneData> Data { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneData
    {
        [JsonProperty("attributes", NullValueHandling = NullValueHandling.Ignore)]
        public HydrophoneAttributes Attributes { get; set; }
        [JsonProperty("id", NullValueHandling = NullValueHandling.Ignore)]
        public string Id { get; set; }
        [JsonProperty("type", NullValueHandling = NullValueHandling.Ignore)]
        public string Type { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneAttributes
    {
        [JsonProperty("image_url", NullValueHandling = NullValueHandling.Ignore)]
        public string ImageUrl { get; set; }
        [JsonProperty("intro_html", NullValueHandling = NullValueHandling.Ignore)]
        public string IntroHtml { get; set; }
        [JsonProperty("location_point", NullValueHandling = NullValueHandling.Ignore)]
        public HydrophoneLocationPoint LocationPoint { get; set; }
        [JsonProperty("name", NullValueHandling = NullValueHandling.Ignore)]
        public string Name { get; set; }
        [JsonProperty("node_name", NullValueHandling = NullValueHandling.Ignore)]
        public string NodeName { get; set; }
        [JsonProperty("slug", NullValueHandling = NullValueHandling.Ignore)]
        public string Slug { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneLocationPoint
    {
        [JsonProperty("coordinates", NullValueHandling = NullValueHandling.Ignore)]
        public List<double> Coordinates { get; set; }
        [JsonProperty("crs", NullValueHandling = NullValueHandling.Ignore)]
        public HydrophoneCrs Crs { get; set; }
        [JsonProperty("type", NullValueHandling = NullValueHandling.Ignore)]
        public string Type { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneCrs
    {
        [JsonProperty("properties", NullValueHandling = NullValueHandling.Ignore)]
        public HydrophoneProperties Properties { get; set; }
        [JsonProperty("type", NullValueHandling = NullValueHandling.Ignore)]
        public string Type { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class HydrophoneProperties
    {
        [JsonProperty("name", NullValueHandling = NullValueHandling.Ignore)]
        public string Name { get; set; }
    }

}

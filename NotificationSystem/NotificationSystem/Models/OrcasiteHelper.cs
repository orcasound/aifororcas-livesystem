using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace NotificationSystem.Models
{
    public class OrcasiteHelper
    {
        private readonly HttpClient _httpClient;
        private string _orcasiteHostname;
        private string _orcasiteApiKey;
        private JsonElement? _orcasiteFeedsArray;
        private readonly ILogger<OrcasiteHelper> _logger;

        public OrcasiteHelper(ILogger<OrcasiteHelper> log, HttpClient httpClient)
        {
            _logger = log;
            _httpClient = httpClient;
        }

        /// <summary>
        /// URI of Orcasite API to post a machine detection.
        /// See https://live.orcasound.net/api/json/swaggerui for details.
        /// </summary>
        public string OrcasitePostDetectionUri
        {
            get
            {
                return "https://" + _orcasiteHostname + "/api/json/detections?fields%5Bdetection%5D=id%2Csource_ip%2Cplaylist_timestamp%2Cplayer_offset%2Clistener_count%2Ctimestamp%2Cdescription%2Cvisible%2Csource%2Ccategory%2Ccandidate_id%2Cfeed_id";
            }
        }

        /// <summary>
        /// URI of Orcasite API to query feeds metadata.
        /// See https://live.orcasound.net/api/json/swaggerui for details.
        /// </summary>
        public string OrcasiteGetFeedsUri
        {
            get
            {
                return "https://" + _orcasiteHostname + "/api/json/feeds?fields%5Bfeed%5D=id%2Cname%2Cnode_name%2Cslug%2Clocation_point%2Cintro_html%2Cimage_url%2Cvisible%2Cbucket%2Cbucket_region%2Ccloudfront_url%2Cdataplicity_id%2Corcahello_id";
            }
        }

        /// <summary>
        /// Get an array of data from a JSON API endpoint.
        /// </summary>
        /// <param name="uri">URI to fetch</param>
        /// <returns>JsonElement representing the JSON array</returns>
        private async Task<JsonElement?> GetDataArrayAsync(string uri)
        {
            try
            {
                string jsonString = await _httpClient.GetStringAsync(uri);
                JsonElement objectElement = JsonSerializer.Deserialize<JsonElement>(jsonString);
                if (objectElement.ValueKind != JsonValueKind.Object)
                {
                    _logger.LogError($"Invalid objectElement kind in ExecuteTask: {objectElement.ValueKind}");
                    return null;
                }
                if (!objectElement.TryGetProperty("data", out var arrayElement))
                {
                    _logger.LogError($"Missing data in ExecuteTask result");
                    return null;
                }
                if (arrayElement.ValueKind != JsonValueKind.Array)
                {
                    _logger.LogError($"Invalid arrayElement kind in ExecuteTask: {arrayElement.ValueKind}");
                    return null;
                }
                return arrayElement;
            }
            catch (Exception e)
            {
                _logger.LogError($"Error: {e.Message}");
                return null;
            }
        }

        /// <summary>
        /// Initialize the OrcasiteHelper object.  Currently this entails
        /// fetching the feeds metadata from Orcasite so that we can convert
        /// a node name to a feed ID.
        ///
        /// TODO: In the future, we could store this info in OrcaHello, or
        /// make the Orcasite API accept OrcaHello feed IDs, and avoid this
        /// extra query.
        /// </summary>
        /// <returns>A task that represents the asynchronous operation</returns>
        public async Task InitializeAsync()
        {
            _orcasiteApiKey = Environment.GetEnvironmentVariable("ORCASITE_APIKEY");
            _orcasiteHostname = Environment.GetEnvironmentVariable("ORCASITE_HOSTNAME") ?? "beta.orcasound.net";
            _orcasiteFeedsArray = await GetDataArrayAsync(OrcasiteGetFeedsUri);
            if (_orcasiteFeedsArray == null)
            {
                _logger.LogError("Failed to retrieve orcasite feeds.");
            }
        }

        /// <summary>
        /// Convert a node name to a feed ID.
        /// </summary>
        /// <param name="nodeNameToFind">Node name to find</param>
        /// <param name="feedsArray">Feeds array to look in</param>
        /// <returns>Feed ID, or null if not found</returns>
        private string GetFeedId(string nodeNameToFind, JsonElement feedsArray)
        {
            foreach (JsonElement feed in feedsArray.EnumerateArray())
            {
                if (!feed.TryGetProperty("attributes", out var attributes))
                {
                    _logger.LogError($"Missing attributes in ExecuteTask result");
                    continue;
                }
                if (attributes.ValueKind != JsonValueKind.Object)
                {
                    _logger.LogError($"Invalid attributes kind in ExecuteTask: {attributes.ValueKind}");
                    continue;
                }
                if (!attributes.TryGetProperty("node_name", out var nodeName))
                {
                    _logger.LogError($"Missing node_name in ExecuteTask result");
                    continue;
                }
                if (nodeName.ValueKind != JsonValueKind.String)
                {
                    _logger.LogError($"Invalid node_name kind in ExecuteTask: {nodeName.ValueKind}");
                    continue;
                }
                string nodeNameString = nodeName.GetString();
                if (nodeNameString != nodeNameToFind)
                {
                    continue;
                }

                if (!feed.TryGetProperty("id", out var feedId))
                {
                    _logger.LogError($"Missing id in ExecuteTask result");
                    continue;
                }
                if (feedId.ValueKind != JsonValueKind.String)
                {
                    _logger.LogError($"Invalid id kind in ExecuteTask: {nodeName.ValueKind}");
                    continue;
                }

                return feedId.GetString();
            }
            return null;
        }

        /// <summary>
        /// Parse JSON representing an OrcaHello detection into just the fields we need.
        /// </summary>
        /// <param name="json">JSON to parse</param>
        /// <param name="id">ID of the detection</param>
        /// <param name="timestampString">Timestamp in the detection</param>
        /// <param name="feedId">Feed ID for the hydrophone node in the detection</param>
        /// <param name="commentsString">Comments in the detection</param>
        /// <returns>true on success, false on failure</returns>
        private bool ParseOrcaHelloDetection(string json, out string idString, out string timestampString, out string feedId, out string commentsString)
        {
            idString = null;
            timestampString = null;
            feedId = null;
            commentsString = null;

            JsonElement orcaHelloDetection = JsonDocument.Parse(json).RootElement;
            if (!orcaHelloDetection.TryGetProperty("id", out var id))
            {
                _logger.LogError($"Missing id in ExecuteTask result");
                return false;
            }
            if (id.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid id kind in ExecuteTask: {id.ValueKind}");
                return false;
            }
            idString = id.GetString();
            if (idString == null)
            {
                _logger.LogError($"Couldn't get ID as a string");
                return false;
            }

            if (!orcaHelloDetection.TryGetProperty("location", out var location))
            {
                _logger.LogError($"Missing location in ExecuteTask result");
                return false;
            }
            if (location.ValueKind != JsonValueKind.Object)
            {
                _logger.LogError($"Invalid location kind in ExecuteTask: {location.ValueKind}");
                return false;
            }

            if (!location.TryGetProperty("id", out var locationId))
            {
                _logger.LogError($"Missing location.id in ExecuteTask result");
                return false;
            }
            if (locationId.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid location.id kind in ExecuteTask: {locationId.ValueKind}");
                return false;
            }
            string locationIdString = locationId.GetString();
            if (locationIdString == null)
            {
                _logger.LogError($"Couldn't get location ID as a string");
                return false;
            }

            // Get feed ID from location ID.
            feedId = GetFeedId(locationIdString, _orcasiteFeedsArray.Value);
            if (feedId == null)
            {
                _logger.LogError($"Couldn't find feed id for: {locationIdString}");
                return false;
            }

            // Get timestamp according to OrcaHello.
            if (!orcaHelloDetection.TryGetProperty("timestamp", out var timestamp))
            {
                _logger.LogError($"Missing timestamp in ExecuteTask result");
                return false;
            }
            if (timestamp.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid timestamp kind in ExecuteTask: {timestamp.ValueKind}");
                return false;
            }
            timestampString = timestamp.GetString();
            if (!DateTime.TryParse(timestampString, out DateTime dateTime))
            {
                _logger.LogError($"Invalid timestamp ExecuteTask: {timestamp}");
                return false;
            }

            // Get comments from OrcaHello.  This property will only be present
            // if the detection was already moderated within OrcaHello.
            if (orcaHelloDetection.TryGetProperty("comments", out var comments))
            {
                commentsString = comments.ValueKind == JsonValueKind.String ? comments.GetString() : null;
            }
            return true;
        }

        /// <summary>
        /// Given an OrcaHello detection (in JSON), report it to Orcasite.
        /// </summary>
        /// <param name="json">OrcaHello detection</param>
        /// <returns>true on success, false on failure</returns>
        public async Task<bool> PostDetectionAsync(string json)
        {
            if (!ParseOrcaHelloDetection(json, out var id, out var timestamp, out var feedId, out var comments))
            {
                return false;
            }

            // We assume that the Azure function lease mechanism ensures that each
            // detection is processed only once, so we don't need to check for duplicates.
            // We also assume that the Orcasite API will reject duplicates if they do occur.

            // Compose a detection post.
            JsonObject newDetection = new JsonObject
            {
                ["data"] = new JsonObject
                {
                    ["type"] = "detection",
                    ["attributes"] = new JsonObject
                    {
                        ["description"] = JsonValue.Create(comments),
                        ["feed_id"] = JsonValue.Create(feedId),
                        ["timestamp"] = JsonValue.Create(timestamp),
                        ["idempotency_key"] = JsonValue.Create(id),

                        // "source": "machine" is implied by a POST.
                        //
                        // "category" is currently disallowed by the API, even though
                        // it would make sense to add "category": "whale" to distinguish
                        // it from, say, a shipnoise detector.
                        //
                        // "playlist_timestamp" and "player_offset" are computed
                        // from the timestamp by the API service so are not needed here.
                    }
                }
            };

            // Try posting it.
            string newDetectionString = newDetection.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
            HttpRequestMessage request = new HttpRequestMessage(HttpMethod.Post, OrcasitePostDetectionUri)
            {
                Content = new StringContent(newDetectionString)
            };
            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/vnd.api+json");
            request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.api+json"));
            if (_orcasiteApiKey != null)
            {
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _orcasiteApiKey);
            }

            HttpResponseMessage response = await _httpClient.SendAsync(request);
            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation($"Detection for {timestamp} posted successfully!");
                return true;
            }
            else
            {
                string message = await response.Content.ReadAsStringAsync();
                _logger.LogError($"Error: {response.StatusCode} - {message}");
                return false;
            }
        }
    }
}

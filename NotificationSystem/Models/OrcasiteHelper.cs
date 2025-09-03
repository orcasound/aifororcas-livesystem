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
        private static HttpClient _httpClient = new HttpClient();
        private string _orcasiteHostname;
        private string? _orcasiteApiKey;
        private JsonElement? _orcasiteFeedsArray;
        private readonly ILogger _logger;

        public OrcasiteHelper(ILogger log)
        {
            _logger = log;
        }

        public string OrcasitePostDetectionUri
        {
            get
            {
                return "https://" + _orcasiteHostname + "/api/json/detections?fields%5Bdetection%5D=id%2Csource_ip%2Cplaylist_timestamp%2Cplayer_offset%2Clistener_count%2Ctimestamp%2Cdescription%2Cvisible%2Csource%2Ccategory%2Ccandidate_id%2Cfeed_id";
            }
        }

        public string OrcasiteGetFeedsUri
        {
            get
            {
                return "https://" + _orcasiteHostname + "/api/json/feeds?fields%5Bfeed%5D=id%2Cname%2Cnode_name%2Cslug%2Clocation_point%2Cintro_html%2Cimage_url%2Cvisible%2Cbucket%2Cbucket_region%2Ccloudfront_url%2Cdataplicity_id%2Corcahello_id";
            }
        }

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

        public async Task InitializeAsync()
        {
            _orcasiteApiKey = Environment.GetEnvironmentVariable("ORCASITE_APIKEY");
            _orcasiteHostname = Environment.GetEnvironmentVariable("ORCASITE_HOSTNAME") ?? "beta.orcasound.net";
            _orcasiteFeedsArray = await GetDataArrayAsync(OrcasiteGetFeedsUri);
            if (_orcasiteFeedsArray == null)
            {
                _logger.LogError("Failed to retrieve orcasite feeds.");
                return;
            }
        }

        private string? GetFeedId(string nodeNameToFind, JsonElement feedsArray)
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

        public async Task PostDetectionAsync(string json)
        {
            JsonElement orcaHelloDetection = JsonDocument.Parse(json).RootElement;
            if (!orcaHelloDetection.TryGetProperty("id", out var id))
            {
                _logger.LogError($"Missing id in ExecuteTask result");
                return;
            }
            if (id.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid id kind in ExecuteTask: {id.ValueKind}");
                return;
            }

            if (!orcaHelloDetection.TryGetProperty("location", out var location))
            {
                _logger.LogError($"Missing location in ExecuteTask result");
                return;
            }
            if (location.ValueKind != JsonValueKind.Object)
            {
                _logger.LogError($"Invalid location kind in ExecuteTask: {location.ValueKind}");
                return;
            }

            if (!location.TryGetProperty("id", out var locationId))
            {
                _logger.LogError($"Missing location.id in ExecuteTask result");
                return;
            }
            if (locationId.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid location.id kind in ExecuteTask: {locationId.ValueKind}");
                return;
            }
            string? locationIdString = locationId.GetString();
            if (locationIdString == null)
            {
                _logger.LogError($"Couldn't get location ID as a string");
                return;
            }

            // Get feed ID from location ID.
            string? feedId = GetFeedId(locationIdString, _orcasiteFeedsArray.Value);
            if (feedId == null)
            {
                _logger.LogError($"Couldn't find feed id for: {locationIdString}");
                return;
            }

            // Get timestamp according to OrcaHello.
            if (!orcaHelloDetection.TryGetProperty("timestamp", out var timestamp))
            {
                _logger.LogError($"Missing timestamp in ExecuteTask result");
                return;
            }
            if (timestamp.ValueKind != JsonValueKind.String)
            {
                _logger.LogError($"Invalid timestamp kind in ExecuteTask: {timestamp.ValueKind}");
                return;
            }
            if (!DateTime.TryParse(timestamp.GetString(), out DateTime dateTime))
            {
                _logger.LogError($"Invalid timestamp ExecuteTask: {timestamp}");
                return;
            }

            // Get comments from OrcaHello.
            if (!orcaHelloDetection.TryGetProperty("comments", out var comments))
            {
                _logger.LogError($"Missing comments in ExecuteTask result");
                return;
            }
            string? commentsString = comments.ValueKind == JsonValueKind.String ? comments.GetString() : null;

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
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _orcasiteApiKey);

            HttpResponseMessage response = await _httpClient.SendAsync(request);

            // Optionally handle response
            if (response.IsSuccessStatusCode)
            {
                _logger.LogInformation($"Detection for {timestamp} posted successfully!");
            }
            else
            {
                string message = await response.Content.ReadAsStringAsync();
                _logger.LogError($"Error: {response.StatusCode} - {message}");
            }
        }
    }
}

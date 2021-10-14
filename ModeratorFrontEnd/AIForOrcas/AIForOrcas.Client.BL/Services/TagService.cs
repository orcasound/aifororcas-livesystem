using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AIForOrcas.Client.BL.Services
{
    public class TagService : ITagService
    {
		private readonly HttpClient httpClient;
		private string api = "api/tags";
		private JsonSerializerOptions defaultJsonSerializerOptions => new JsonSerializerOptions() { PropertyNameCaseInsensitive = true };

		public TagService(HttpClient httpClient)
		{
			this.httpClient = httpClient;
		}

		// Get detections based on passed view, pagination options, and filter options
		public async Task<List<string>> GetUniqueTagsAsync()
		{
			var httpResponseMessage = await httpClient.GetAsync(api);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
				var responseString = await httpResponseMessage.Content.ReadAsStringAsync();

				if (string.IsNullOrWhiteSpace(responseString))
					return new List<string>();

				return JsonSerializer.Deserialize<List<string>>(responseString, defaultJsonSerializerOptions);
			}
			else
			{
				return new List<string>();
			}
		}
	}
}

using Orcasound.Shared.Entities;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace Orcasound.UI.Services
{
	public class APICandidateService : ICandidateService
	{
		private readonly HttpClient httpClient;
		private string url = "api/aiclipmetadata";
		private JsonSerializerOptions defaultJsonSerializerOptions =>
			new JsonSerializerOptions() { PropertyNameCaseInsensitive = true };

		public APICandidateService(HttpClient httpClient)
		{
			this.httpClient = httpClient;
		}

		public async Task<PaginatedResponse<List<Candidate>>> GetCandidates(Pagination pagination)
		{
			var newURL = url.Contains("?") ?
				$"{url}&page={pagination.Page}&recordsPerPage={pagination.RecordsPerPage}&sortBy={pagination.SortBy}&sortOrder={pagination.SortOrder}&timeframe={pagination.Timeframe}" :
				$"{url}?page={pagination.Page}&recordsPerPage={pagination.RecordsPerPage}&sortBy={pagination.SortBy}&sortOrder={pagination.SortOrder}&timeframe={pagination.Timeframe}";

			var httpResponseMessage = await httpClient.GetAsync(newURL);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
				var responseString = await httpResponseMessage.Content.ReadAsStringAsync();
				var response = JsonSerializer.Deserialize<List<Candidate>>(responseString, defaultJsonSerializerOptions);

				return new PaginatedResponse<List<Candidate>> { Response = response, 
					TotalAmountPages = int.Parse(httpResponseMessage.Headers.GetValues("totalAmountPages").FirstOrDefault()),
					TotalNumberRecords = int.Parse(httpResponseMessage.Headers.GetValues("totalNumberRecords").FirstOrDefault())};
			}
			else
			{
				return new PaginatedResponse<List<Candidate>> { Response = new List<Candidate>(), TotalAmountPages = 0, TotalNumberRecords = 0 };
			}
		}

		public async Task<PaginatedResponse<List<Candidate>>> GetUnreviewedCandidates(Pagination pagination)
		{
			var newURL = url.Contains("?") ?
				$"{url}/unreviewed&page={pagination.Page}&recordsPerPage={pagination.RecordsPerPage}&sortBy={pagination.SortBy}&sortOrder={pagination.SortOrder}&timeframe={pagination.Timeframe}" :
				$"{url}/unreviewed?page={pagination.Page}&recordsPerPage={pagination.RecordsPerPage}&sortBy={pagination.SortBy}&sortOrder={pagination.SortOrder}&timeframe={pagination.Timeframe}";

			var httpResponseMessage = await httpClient.GetAsync(newURL);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
				var responseString = await httpResponseMessage.Content.ReadAsStringAsync();
				var response = JsonSerializer.Deserialize<List<Candidate>>(responseString, defaultJsonSerializerOptions);

				return new PaginatedResponse<List<Candidate>>
				{
					Response = response,
					TotalAmountPages = int.Parse(httpResponseMessage.Headers.GetValues("totalAmountPages").FirstOrDefault()),
					TotalNumberRecords = int.Parse(httpResponseMessage.Headers.GetValues("totalNumberRecords").FirstOrDefault())
				};
			}
			else
			{
				return new PaginatedResponse<List<Candidate>> { Response = new List<Candidate>(), TotalAmountPages = 0, TotalNumberRecords = 0 };
			}
		}

		public async Task UpdateCandidate(UpdateRequest request)
		{
			var newUrl = $"{url}/{request.Id}";
			var dataJson = JsonSerializer.Serialize(request);
			var stringContent = new StringContent(dataJson, Encoding.UTF8, "application/json");
			var httpResponseMessage = await httpClient.PutAsync(newUrl, stringContent);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
			}
		}
	}
}

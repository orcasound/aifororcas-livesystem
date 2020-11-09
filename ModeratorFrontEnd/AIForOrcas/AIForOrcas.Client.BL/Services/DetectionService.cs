using AIForOrcas.DTO;
using AIForOrcas.DTO.API;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AIForOrcas.Client.BL.Services
{
	public class DetectionService : IDetectionService
	{
		private readonly HttpClient httpClient;
		private string api = "api/detections";
		private JsonSerializerOptions defaultJsonSerializerOptions => new JsonSerializerOptions() { PropertyNameCaseInsensitive = true };

		public DetectionService(HttpClient httpClient)
		{
			this.httpClient = httpClient;
		}

		// Get detections based on passed view, pagination options, and filter options
		private async Task<PaginatedResponseDTO<List<Detection>>> GetDetectionsAsync(string viewName, PaginationOptionsDTO paginationOptions, IFilterOptions filterOptions)
		{
			var prefix = api.Contains("?") ? $"{api}/{viewName}&" : $"{api}/{viewName}?";
			var url = $"{prefix}{paginationOptions.QueryString}&{filterOptions.QueryString}";

			var httpResponseMessage = await httpClient.GetAsync(url);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
				var responseString = await httpResponseMessage.Content.ReadAsStringAsync();

				if(string.IsNullOrWhiteSpace(responseString))
					return new PaginatedResponseDTO<List<Detection>> { Response = new List<Detection>(), TotalAmountPages = 0, TotalNumberRecords = 0 };

				return new PaginatedResponseDTO<List<Detection>>
				{
					Response = JsonSerializer.Deserialize<List<Detection>>(responseString, defaultJsonSerializerOptions),
					TotalAmountPages = int.Parse(httpResponseMessage.Headers.GetValues("totalAmountPages").FirstOrDefault()),
					TotalNumberRecords = int.Parse(httpResponseMessage.Headers.GetValues("totalNumberRecords").FirstOrDefault())
				};
			}
			else
			{
				return new PaginatedResponseDTO<List<Detection>> { Response = null, TotalAmountPages = 0, TotalNumberRecords = 0 };
			}

		}

		// Get unreviewed detections
		public async Task<PaginatedResponseDTO<List<Detection>>> GetCandidateDetectionsAsync(PaginationOptionsDTO paginationOptions, IFilterOptions filterOptions)
		{
			return await GetDetectionsAsync("unreviewed", paginationOptions, filterOptions);
		}

		public async Task<PaginatedResponseDTO<List<Detection>>> GetConfirmedDetectionsAsync(PaginationOptionsDTO paginationOptions, IFilterOptions filterOptions)
		{
			return await GetDetectionsAsync("confirmed", paginationOptions, filterOptions);
		}

		public async Task<PaginatedResponseDTO<List<Detection>>> GetFalseDetectionsAsync(PaginationOptionsDTO paginationOptions, IFilterOptions filterOptions)
		{
			return await GetDetectionsAsync("falsepositives", paginationOptions, filterOptions);
		}

		public async Task<PaginatedResponseDTO<List<Detection>>> GetUnconfirmedDetectionsAsync(PaginationOptionsDTO paginationOptions, IFilterOptions filterOptions)
		{
			return await GetDetectionsAsync("unknowns", paginationOptions, filterOptions);
		}

		public async Task UpdateRequestAsync(DetectionUpdate request)
		{
			var url = $"{api}/{request.Id}";
			var dataJson = JsonSerializer.Serialize(request);
			var stringContent = new StringContent(dataJson, Encoding.UTF8, "application/json");

			var httpResponseMessage = await httpClient.PutAsync(url, stringContent);
		}

		public async Task<Detection> GetDetectionAsync(string id)
		{
			var url = $"{api}/{id}";

			var httpResponseMessage = await httpClient.GetAsync(url);

			if (httpResponseMessage.IsSuccessStatusCode)
			{
				var responseString = await httpResponseMessage.Content.ReadAsStringAsync();

				var response = JsonSerializer.Deserialize<Detection>(responseString, defaultJsonSerializerOptions);

				return response;
			}
			else
			{
				return new Detection();
			}
		}
	}
}

using AIForOrcas.DTO;
using Microsoft.AspNetCore.Components;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Components
{
	public partial class CandidateFilterComponent
	{
		[Parameter]
		public CandidateFilterOptionsDTO FilterOptions { get; set; } = new CandidateFilterOptionsDTO();

		[Parameter]
		public EventCallback<CandidateFilterOptionsDTO> ApplyFilterCallback { get; set; }

		private async Task ApplyFilter()
		{
			await ApplyFilterCallback.InvokeAsync(FilterOptions);
		}
	}
}

using AIForOrcas.DTO;
using Microsoft.AspNetCore.Components;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Components
{
	public partial class ReviewedFilterComponent
	{
		[Parameter]
		public ReviewedFilterOptionsDTO FilterOptions { get; set; } = new ReviewedFilterOptionsDTO();

		[Parameter]
		public EventCallback<ReviewedFilterOptionsDTO> ApplyFilterCallback { get; set; }

		private async Task ApplyFilter()
		{
			await ApplyFilterCallback.InvokeAsync(FilterOptions);
		}
	}
}

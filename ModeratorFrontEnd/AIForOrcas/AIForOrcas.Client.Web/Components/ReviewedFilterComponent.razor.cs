using AIForOrcas.DTO;
using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.Configuration;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Components
{
	public partial class ReviewedFilterComponent
	{
		[Parameter]
		public ReviewedFilterOptionsDTO FilterOptions { get; set; } = new ReviewedFilterOptionsDTO();

		[Parameter]
		public EventCallback<ReviewedFilterOptionsDTO> ApplyFilterCallback { get; set; }

		[Inject]
		public IConfiguration Configuration { get; set; }

		private List<string> AllLocations = new List<string>();

		protected override void OnInitialized()
		{
			AllLocations = Configuration.GetSection("Locations").Get<List<string>>();
		}

		private async Task ApplyFilter()
		{
			await ApplyFilterCallback.InvokeAsync(FilterOptions);
		}
	}
}

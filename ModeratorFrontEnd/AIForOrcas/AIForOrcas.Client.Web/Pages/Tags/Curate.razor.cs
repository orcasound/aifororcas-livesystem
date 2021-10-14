using AIForOrcas.Client.BL.Services;
using Blazored.Toast.Services;
using Microsoft.AspNetCore.Components;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Pages.Tags
{
    public partial class Curate : ComponentBase
    {
        [Inject]
        ITagService Service { get; set; }

        [Inject]
        IToastService ToastService { get; set; }

        private List<string> tags = null;

        private int totalNumberOfTags = 0;

		private string loadStatus = null;

		protected override async Task OnInitializedAsync()
		{
			await LoadTags();
		}

		private async Task LoadTags()
		{
			loadStatus = "Loading records...";
			tags = await Service.GetUniqueTagsAsync();

			loadStatus = null;
			totalNumberOfTags = tags.Count();
		}
	}
}

using AIForOrcas.Client.BL.Services;
using AIForOrcas.DTO.API.Tags;
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
		private bool showDeleteModal = false;
		private string tagToDelete = string.Empty;
		private bool showEditModal = false;
		private TagUpdate tagUpdate = null;

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

		private void OpenEditModal(string tag)
        {
			tagUpdate = new TagUpdate
			{
				OldTag = tag,
				NewTag = tag
			};

			showEditModal = true;
        }

		private void CloseEditModal()
        {
			tagUpdate = null;
			showEditModal = false;
        }

		private async Task HandleValidEditSubmit()
        {
			var result = await Service.UpdateTagAsync(tagUpdate);

			ToastService.ShowSuccess($"{result} record(s) successfully updated.");

			tagUpdate = null;

			showEditModal = false;

			await LoadTags();
		}

		private void OpenDeleteModal(string tag)
        {
			tagToDelete = tag;
			showDeleteModal = true;
        }

		private async Task HandleDelete()
        {
			var result = await Service.DeleteTagAsync(tagToDelete);

			ToastService.ShowSuccess($"{result} record(s) successfully updated.");

			tagToDelete = string.Empty;
			showDeleteModal = false;

			await LoadTags();
		}

		private void CloseDeleteModal()
        {
			tagToDelete = string.Empty;
			showDeleteModal = false;
        }
	}
}

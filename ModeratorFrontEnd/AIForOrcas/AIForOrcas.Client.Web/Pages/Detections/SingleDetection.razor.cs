using AIForOrcas.Client.BL.Services;
using AIForOrcas.DTO.API;
using Blazored.Toast.Services;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using System;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Pages.Detections
{
	public partial class SingleDetection : IDisposable
	{
		[Parameter]
		public string Id { get; set; }

		[Inject]
		IJSRuntime JSRuntime { get; set; }

		[Inject]
		IDetectionService Service { get; set; }

		[Inject]
		IToastService ToastService { get; set; }

		private Detection detection = null;
		private bool isFound = true;
		private bool isEditable = false;

		protected override async Task OnInitializedAsync()
		{
			await LoadDetection();
		}

		private async Task LoadDetection()
		{
			detection = await Service.GetDetectionAsync(Id);
			if(detection.Id == null)
				isFound = false;

			if(detection.Found.ToLower() == "don't know" || !detection.Reviewed)
				isEditable = true;
		}

		private async Task ActOnSubmitCallback(DetectionUpdate request)
		{
			await Service.UpdateRequestAsync(request);

			ToastService.ShowSuccess("Detection successfully updated.");

			await LoadDetection();
		}

		void IDisposable.Dispose()
		{
			JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
		}
	}
}

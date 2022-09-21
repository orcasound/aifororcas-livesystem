namespace AIForOrcas.Client.Web.Pages.Detections;

public partial class SingleDetection : ComponentBase, IDisposable
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

	protected override async Task OnInitializedAsync()
	{
		await LoadDetection();
	}

	private async Task LoadDetection()
	{
		detection = await Service.GetDetectionAsync(Id);
		if(detection.Id == null)
			isFound = false;
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

namespace AIForOrcas.Client.Web.Pages.Detections;

public partial class Candidates : IDisposable
{
	[Inject]
	IJSRuntime JSRuntime { get; set; }

	[Inject]
	IDetectionService Service { get; set; }

	[Inject]
	IToastService ToastService { get; set; }

	private List<Detection> detections = null;

	private PaginationOptionsDTO paginationOptions = 
		new PaginationOptionsDTO() { RecordsPerPage = 5, Page = 1 };

	private CandidateFilterOptionsDTO filterOptions =
		new CandidateFilterOptionsDTO() { SortBy = "timestamp", SortOrder = "desc", Timeframe = "6h", Location = "all", HydrophoneId = "all" };

	private PaginationResultsDTO pagination = new PaginationResultsDTO();

	private string loadStatus = null;

	protected override async Task OnInitializedAsync()
	{
		await LoadDetections();
	}

	private async Task LoadDetections()
	{
		loadStatus = "Loading records...";
		var paginatedResponse = await Service.GetCandidateDetectionsAsync(paginationOptions, filterOptions);

		pagination.TotalNumberOfRecords = paginatedResponse.TotalNumberRecords;
		pagination.TotalNumberOfPages = paginatedResponse.TotalAmountPages;
		pagination.CurrentPage = paginationOptions.Page;

		if (paginatedResponse.Response == null)
			loadStatus = "An unknown error occurred while loading records...";
		else if (paginatedResponse.Response.Count == 0)
			loadStatus = "No records found for the selected filter options. Please select a different set of filter options...";
		else
		{
			loadStatus = null;
			detections = paginatedResponse.Response;
		}
	}

	private async Task ActOnSelectPageCallback(PaginationOptionsDTO returnedPaginationOptions)
	{
		paginationOptions = returnedPaginationOptions;
		detections = null;
		await LoadDetections();
		await JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
		StateHasChanged();
	}

	private async Task ActOnApplyFilterCallback(CandidateFilterOptionsDTO returnedFilterOptions)
	{
		filterOptions = returnedFilterOptions;
		paginationOptions.Page = 1;
		detections = null;
		await LoadDetections();
		await JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
		StateHasChanged();
	}

	private async Task ActOnSubmitCallback(DetectionUpdate request)
	{
		await Service.UpdateRequestAsync(request);

		ToastService.ShowSuccess("Detection successfully updated.");

		paginationOptions.Page = 1;
		await LoadDetections();
	}

	void IDisposable.Dispose()
	{
		JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
	}

}

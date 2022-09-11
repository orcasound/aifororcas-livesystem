namespace AIForOrcas.Client.Web.Components;

public partial class CandidateFilterComponent
{
	[Parameter]
	public CandidateFilterOptionsDTO FilterOptions { get; set; } = new CandidateFilterOptionsDTO();

	[Parameter]
	public EventCallback<CandidateFilterOptionsDTO> ApplyFilterCallback { get; set; }

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

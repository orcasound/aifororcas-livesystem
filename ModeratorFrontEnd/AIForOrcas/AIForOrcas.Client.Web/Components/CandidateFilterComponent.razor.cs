namespace AIForOrcas.Client.Web.Components;

public partial class CandidateFilterComponent
{
	[Parameter]
	public CandidateFilterOptionsDTO FilterOptions { get; set; } = new CandidateFilterOptionsDTO();

	[Parameter]
	public EventCallback<CandidateFilterOptionsDTO> ApplyFilterCallback { get; set; }

	[Inject]
	public AppSettings AppSettings { get; set; }

	private List<string> AllLocations = new List<string>();

	protected override void OnInitialized()
	{
		AllLocations = AppSettings.Locations.ToList();
	}

	private async Task ApplyFilter()
	{
		// The HydrophoneId (e.g., rpi_orcasound_lab) is constant whereas the Location
		// value can and has changed over time (e.g., Haro Strait vs Orcasound Lab).
		// The UI lets the user choose among labels that are Location values, but
		// we want to actually query by HydrophoneId.
		if (FilterOptions.Location != "all")
		{
			int index = Array.IndexOf(AppSettings.Locations, FilterOptions.Location);
			if (index >= 0 && index < AppSettings.HydrophoneIds.Length)
			{
				FilterOptions.Location = "all";
				FilterOptions.HydrophoneId = AppSettings.HydrophoneIds[index];
			}
		}
		else
		{
			FilterOptions.HydrophoneId = "all";
		}

		await ApplyFilterCallback.InvokeAsync(FilterOptions);
	}
}

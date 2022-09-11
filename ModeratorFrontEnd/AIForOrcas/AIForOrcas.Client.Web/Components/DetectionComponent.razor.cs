namespace AIForOrcas.Client.Web.Components;

public partial class DetectionComponent
{
	private string _id;
	private TextInfo _ti = new CultureInfo("en-US", false).TextInfo;

	[Inject]
	IJSRuntime JSRuntime { get; set; }

	[Inject]
	AuthenticationStateProvider AuthenticationStateProvider { get; set; }

	[Inject]
	NavigationManager NavigationManager { get; set; }

	[Parameter]
	public Detection Detection { get; set; }

	[Parameter]
	public bool IsEditable { get; set; } = true;

	[Parameter]
	public EventCallback<DetectionUpdate> SubmitCallback { get; set; }

	private string[] optionList = new string[] { "Yes", "No", "Don't Know" };

	private string CardSpectrogramId { get => $"spectrogram-card-{_id}"; }
	private string CardWaveformId { get => $"waveform-card-{_id}"; }
	private string CardPlayButtonId { get => $"play-card-{_id}"; }
	private string CardElapsedTimeId { get => $"elapsed-card-{_id}"; }
	private string CardDurationTimeId { get => $"duration-card-{_id}"; }

	private string ModalSpectrogramPanelId { get => $"spectrogram-panel-modal-{_id}"; }
	private string ModalSpectrogramId { get => $"spectrogram-modal-{_id}"; }
	private string ModalWaveformId { get => $"waveform-modal-{_id}"; }
	private string ModalPlayButtonId { get => $"play-modal-{_id}"; }
	private string ModalElapsedTimeId { get => $"elapsed-modal-{_id}"; }
	private string ModalDurationTimeId { get => $"duration-modal-{_id}"; }

	private string ModalMapPanelId { get => $"map-panel-modal-{_id}"; }
	private string BingMapId { get => $"bingMap-modal-{_id}"; }

	private string ModalLinkId { get => $"link-panel-modal-{_id}"; }

	private string DetectionCount { get => (Detection.Annotations.Count == 1) ? "1 detection" : $"{Detection.Annotations.Count} detections"; }

	private string AverageConfidence { get => $"{Detection.Confidence.ToString("00.##")}% average confidence"; }

	private bool IsSubmitDisabled { get => string.IsNullOrWhiteSpace(Detection.Found); }

	private string WasFound	{ get => _ti.ToTitleCase(Detection.Found); }

	private string LinkUrl { get => $"{NavigationManager.BaseUri}detections/detection/{Detection.Id}"; }

	protected override void OnParametersSet()
	{
		_id = Detection.Id;

		// Unreviewed detections are being initially populated in the database as "No"
		// I am manually resetting it here when the reviewed status is false so that the record,
		// can be unsubmittable until the user has changed Found to "Yes", "No", or "Don't Know"

		// TODO: Determine whether or not we should change the initial Found state
		//       from No to something other than the three options we give the user

		if (!Detection.Reviewed)
			Detection.Found = string.Empty;
	}

	private void SetFoundValue(string found)
	{
		Detection.Found = found;
	}

	private async Task SubmitUpdate()
	{
		var authState = await AuthenticationStateProvider.GetAuthenticationStateAsync();
		var user = authState.User;

		var request = new DetectionUpdate()
		{
			Id = Detection.Id,
			Comments = Detection.Comments,
			Tags = Detection.Tags,
			Moderator = user.Identity.Name,
			Moderated = DateTime.Now,
			Reviewed = true,
			Found = Detection.Found
		};

		await SubmitCallback.InvokeAsync(request);
	}

	private async Task ToggleCardPlayer()
	{
		await JSRuntime.InvokeVoidAsync("CardSpectrogram", _id, Detection.AudioUri);
	}

	private async Task ToggleModalPlayer()
	{
		var isPlaying = await JSRuntime.InvokeAsync<bool>("IsPlayerActive");

		if (!isPlaying)
		{
			await InitializeModalPlayer();
		}

		await JSRuntime.InvokeVoidAsync("ToggleModalSpectrogram");
	}

	private async Task InitializeModalPlayer()
	{
		StringBuilder sb = new StringBuilder();
		sb.Append("[");

		var list = new List<string>();
		foreach(var annotation in Detection.Annotations)
		{
			var entry = "{";
			entry += $"\"start\":{annotation.StartTime}, ";
			entry += $"\"end\":{annotation.EndTime},";
			entry += "\"drag\": false";
			entry += "}";
			list.Add(entry);
		}
		sb.Append(string.Join(",", list));

		sb.AppendLine("]");

		await JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
		await JSRuntime.InvokeVoidAsync("InitializeModalSpectrogram", _id, 
			Detection.AudioUri, sb.ToString());
	}

	private async Task InitializeModalMap()
	{
		await JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
		await JSRuntime.InvokeVoidAsync("LoadBingMap", _id, 
			Detection.Location?.Latitude, Detection.Location?.Longitude);
	}

	private async Task KillPlayer()
	{
		await JSRuntime.InvokeVoidAsync("DestroyActivePlayer");
	}

	private async Task ActivateLink(string url)
	{
		var authState = await AuthenticationStateProvider.GetAuthenticationStateAsync();

		NavigationManager.NavigateTo(url, true);
	}
}

using AIForOrcas.Client.BL.Helpers;
using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Components
{
	public partial class TopBarComponent
	{
		[Inject]
		IJSRuntime JSRuntime { get; set; }

		[Inject]
		IdentityHelper IdentityHelper { get; set; }

		[Parameter]
		public string CurrentUrl { get; set; }

		private string UserName { get; set; }

		private string DisplayDate { get; set; }

		private string ShortDisplayDate { get; set; }

		public CancellationTokenSource CancellationTokenSource { get; set; }

		[Parameter]
		public EventCallback ToggleThemeCallback { get; set; }

		private string theme = "Dark";

		private void SetDateTime()
		{
			var now = DateTime.UtcNow;
			DisplayDate = DateHelper.UTCToPDT(now);
			ShortDisplayDate = DateHelper.UTCToPDT(now, true);
		}

		protected override async Task OnInitializedAsync()
		{
			UserName = await IdentityHelper.GetName();
			SetDateTime();
			CancellationTokenSource = new CancellationTokenSource();
			await RealTimeUpdate(CancellationTokenSource.Token);
		}

		private async Task ToggleTheme()
		{
			theme = (theme == "Dark") ? "Light" : "Dark";
			await ToggleThemeCallback.InvokeAsync(null);
		}

		private async Task ToggleSidebar()
		{
			await JSRuntime.InvokeVoidAsync("ToggleSideBar");
		}

		public async Task RealTimeUpdate(CancellationToken cancellationToken)
		{
			while(!cancellationToken.IsCancellationRequested)
			{
				await Task.Delay(1000, cancellationToken);
				if (!cancellationToken.IsCancellationRequested)
				{
					SetDateTime();
					await InvokeAsync(() => this.StateHasChanged());
				}
			}
		}
	}
}

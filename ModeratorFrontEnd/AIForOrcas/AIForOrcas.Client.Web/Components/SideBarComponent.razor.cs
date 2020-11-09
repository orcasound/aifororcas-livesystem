using Microsoft.AspNetCore.Components;
using Microsoft.JSInterop;
using System.Threading.Tasks;

namespace AIForOrcas.Client.Web.Components
{
	public partial class SideBarComponent
	{
		[Inject]
		IJSRuntime JSRuntime { get; set; }
		
		private async Task ToggleDisplay()
		{
			await JSRuntime.InvokeVoidAsync("ToggleSideBar");
		}
	}
}

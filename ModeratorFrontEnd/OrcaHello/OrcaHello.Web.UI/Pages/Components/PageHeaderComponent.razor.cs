﻿namespace OrcaHello.Web.UI.Pages.Components
{
    public partial class PageHeaderComponent
    {
        [Parameter]
        public string Title { get; set; } = string.Empty;

        [Parameter]
        public string Description { get; set; } = string.Empty;
    }
}
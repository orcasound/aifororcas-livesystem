﻿namespace OrcaHello.Web.UI.Services
{
    public interface ITagService
    {
        ValueTask<List<string>> RetrieveAllTagsAsync();
        ValueTask<TagListForTimeframeResponse> RetrieveFilteredTagsAsync(DateTime? fromDate, DateTime? toDate);
    }
}
namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public static class DropdownHelper
    {
        public static List<DropdownOption> DetectionStateDropdownOptions
        {
            get
            {
                return new List<DropdownOption>
                {
                    new(DetectionState.Positive.ToString(), "Yes"),
                    new(DetectionState.Negative.ToString(), "No"),
                    new(DetectionState.Unknown.ToString(), "Don't Know")
                };
            }
        }
    }
}

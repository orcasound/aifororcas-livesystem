namespace OrcaHello.Web.UI
{
    [ExcludeFromCodeCoverage]
    public class DropdownOption
    {
        public string Text { get; private set; }
        public object Value { get; private set; }

        public DropdownOption(object value, string text)
        {
            Value = value;
            Text = text;
        }
    }
}

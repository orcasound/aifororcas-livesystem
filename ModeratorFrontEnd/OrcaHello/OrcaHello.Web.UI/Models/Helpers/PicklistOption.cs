namespace OrcaHello.Web.UI.Models
{
    public class PicklistOption
    {
        public string Text { get; private set; }
        public int Value { get; private set; }
        public string ValueText { get; private set; }
        public string Icon { get; set; }

        public PicklistOption(int value, string text, string valueText = null!, string icon = null!)
        {
            Value = value;
            Text = text;
            ValueText = valueText;
            Icon = icon;
        }
    }
}

namespace OrcaHello.Web.UI.Models.Dashboard
{
    public class MetricsItemView
    {
        public string Name { get; set; } = string.Empty;
        public int Value { get; set; } = 0;

        public MetricsItemView(string name, int value)
        {
            Name = name;
            Value = value;
        }
    }
}

namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public static class DateTimeExtensions
    {
        public static DateTime Adjust(this DateTime dateTime, Timeframe timeframe)
        {
            switch (timeframe)
            {
                case Timeframe.ThirtyMinutes:
                    return dateTime.AddMinutes(-30);
                case Timeframe.ThreeHours:
                    return dateTime.AddHours(-3);
                case Timeframe.SixHours:
                    return dateTime.AddHours(-6);
                case Timeframe.TwentyFourHours:
                    return dateTime.AddHours(-24);
                case Timeframe.SevenDays:
                    return dateTime.AddDays(-7);
                case Timeframe.ThirtyDays:
                    return dateTime.AddDays(-30);
                case Timeframe.ThreeMonths:
                    return dateTime.AddMonths(-3);
                case Timeframe.SixMonths:
                    return dateTime.AddMonths(-6);
                case Timeframe.OneYear:
                    return dateTime.AddYears(-1);
                default:
                    throw new ArgumentOutOfRangeException(nameof(Timeframe), timeframe, null);
            }
        }
    }
}

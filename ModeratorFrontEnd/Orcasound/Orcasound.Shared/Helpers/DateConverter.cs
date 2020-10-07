using System;

namespace Orcasound.Shared.Helpers
{
	public static class DateConverter
	{
		public static string ToPDT(DateTime datetime)
		{
			TimeZoneInfo pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
			DateTime pstTime = TimeZoneInfo.ConvertTimeFromUtc(TimeZoneInfo.ConvertTimeToUtc(datetime), pstZone);
			var zoneString = pstZone.IsDaylightSavingTime(pstTime) ? "PDT" : "PST";
			return $"{pstTime} {zoneString}";
		}
	}
}

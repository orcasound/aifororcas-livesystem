using System;

namespace AIForOrcas.Client.BL.Helpers
{
	public static class DateHelper
	{
		public static string ToPDT(DateTime datetime, bool timeOnly = false)
		{
			TimeZoneInfo pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
			DateTime pstTime = TimeZoneInfo.ConvertTimeFromUtc(TimeZoneInfo.ConvertTimeToUtc(datetime), pstZone);
			var zoneString = pstZone.IsDaylightSavingTime(pstTime) ? "PDT" : "PST";
			var format = timeOnly ? $"HH:mm:ss" : $"dd MMM HH:mm:ss '{zoneString}'";
			return $"{pstTime.ToString(format)}";
		}

		public static string ToPDTFull(DateTime datetime)
		{
			TimeZoneInfo pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
			DateTime pstTime = TimeZoneInfo.ConvertTimeFromUtc(TimeZoneInfo.ConvertTimeToUtc(datetime), pstZone);
			var zoneString = pstZone.IsDaylightSavingTime(pstTime) ? "PDT" : "PST";
			var format = $"dd MMM yyyy HH:mm:ss '{zoneString}'";
			return $"{pstTime.ToString(format)}";
		}
	}
}

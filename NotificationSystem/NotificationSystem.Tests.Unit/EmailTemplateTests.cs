using Newtonsoft.Json.Linq;
using NotificationSystem.Template;
using System;
using System.Collections.Generic;

namespace NotificationSystem.Tests.Unit
{
    public class EmailTemplateTests
    {
        /// <summary>
        /// Tests that GetSubscriberEmailBody generates correct map URIs for various locations
        /// by verifying the generated HTML contains the expected image URLs.
        /// </summary>
        [Theory]
        [InlineData("Sunset Bay", "sunset-bay.jpg")]
        [InlineData("Mast Center", "mast-center.jpg")]
        [InlineData("North San Juan Channel", "north-san-juan-channel.jpg")]
        [InlineData("Point Robinson", "point-robinson.jpg")]
        [InlineData("Bush Point", "bush-point.jpg")]
        [InlineData("Haro Strait", "haro-strait.jpg")]
        [InlineData("Port Townsend", "port-townsend.jpg")]
        [InlineData("Orcasound Lab", "orcasound-lab.jpg")]
        public void GetSubscriberEmailBody_GeneratesCorrectMapUri_ForLocationName(string locationName, string expectedFileName)
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = locationName,
                        latitude = 48.123,
                        longitude = -122.456,
                        id = "test_location"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            string expectedMapUrl = $"https://orcanotificationstorage.blob.core.windows.net/images/{expectedFileName}";

            // Act
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages);

            // Assert
            Assert.Contains(expectedMapUrl, emailBody);
        }

        /// <summary>
        /// Tests that location names with multiple words are correctly converted with hyphens in URIs.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_HandlesMultipleSpacesCorrectly()
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = "North San Juan Channel",
                        latitude = 48.591294,
                        longitude = -123.058779,
                        id = "rpi_north_sjc"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            // Act
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages);

            // Assert - the URI should use hyphens
            Assert.Contains("north-san-juan-channel.jpg", emailBody);
            // The location name should still display with spaces
            Assert.Contains("North San Juan Channel", emailBody);
        }

        /// <summary>
        /// Tests that the email body contains all required sections and location information.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_ContainsAllRequiredSections()
        {
            // Arrange
            var testTimestamp = new DateTime(2025, 1, 15, 10, 30, 0, DateTimeKind.Utc);
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = testTimestamp,
                    location = new
                    {
                        name = "Sunset Bay",
                        latitude = 47.86497296593844,
                        longitude = -122.33393605795372,
                        id = "rpi_sunset_bay"
                    },
                    moderator = "Jane Doe",
                    comments = "Clear SRKW calls detected"
                })
            };

            // Act
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages);

            // Assert
            Assert.Contains("Southern Resident Killer Whale Detected", emailBody);
            Assert.Contains("Sunset Bay", emailBody);
            Assert.Contains("47.86497296593844", emailBody);
            Assert.Contains("-122.33393605795372", emailBody);
            Assert.Contains("Jane Doe", emailBody);
            Assert.Contains("Clear SRKW calls detected", emailBody);
            Assert.Contains("https://orcanotificationstorage.blob.core.windows.net/images/sunset-bay.jpg", emailBody);
        }
    }
}

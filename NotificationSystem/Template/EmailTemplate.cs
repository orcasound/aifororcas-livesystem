using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;

namespace NotificationSystem.Template
{
    // TODO: we should move all html out of code and maybe use a preset email template for better design. 
    public static class EmailTemplate
    {
        public static string GetModeratorEmailBody(DateTime? timestamp, string location)
        {
            return $"<html><head><style>{GetCSS()}</style></head><body>{GetModeratorEmailHtml(timestamp, location)}</body></html>";
        }

        public static string GetSubscriberEmailBody(List<JObject> messages)
        {
            return $"<html><head><style>{GetCSS()}</style></head><body>{GetSubscriberEmailHtml(messages)}</body></html>";
        }

        private static string GetSubscriberEmailHtml(List<JObject> messages)
        {
            string timeString = GetPDTTimestring((DateTime?) messages[0]["timestamp"]);

            return $@"
                <body>
                <div class='card'>
                <h1>
                Southern Resident Killer Whale Detected
                </h1>
                <p>
                Dear subscriber, a Southern Resident Killer Whale was most recently detected at around {timeString} PDT. 
                </p>
                <p>
                Please be mindful of their presence when travelling in the areas below.
                </p>
                <hr/>
                <h2>
                Detections
                </h2>
                <p>
                <center>
                  <table style='width:70%;'>
                  {GetDetectedSectionHtml(messages)}
                  </table>
                </center>
                </p>
                </div>
                <footer>
                  In partnership with Microsoft AI 4 Earth, Orcasound and Orca Conservancy.
                </footer>
                </body>
            ";
        }

        private static string GetDetectedSectionHtml(List<JObject> messages)
        {
            string rows = "";
            foreach (JObject message in messages)
            {
                string timeString = GetPDTTimestring((DateTime?)message["timestamp"]);

                rows += $@"
                  <tr>
                    <td>
                      <img src='{GetMapUri((string) message["location"]["name"])}'>
                    </td>
                    <td>
                      <ul>
                      <li><b>Time Detected:</b> {timeString} </li>
                      <li><b>Location:</b> {message["location"]["name"]} - {message["location"]["latitude"]}, {message["location"]["longitude"]} </li>
                      <li><b>Moderated By: </b> {message["moderator"]} </li>
                      </ul>
                      <p>
                        <center>
                          <b>Moderator Comments:</b><br>
                          {message["comments"]}
                        </center>
                      </p>
                    </td>
                  </tr>
                ";
            }

            return rows;
        }

        private static string GetMapUri(string locationName)
        {
            if (locationName.ToLower() == "haro strait")
            {
                return "https://orcanotificationstorage.blob.core.windows.net/images/haropoint.jpg";
            }
            else if (locationName.ToLower() == "bush point")
            {
                return "https://orcanotificationstorage.blob.core.windows.net/images/bushpoint.jpg";
            }
            else
            {
                return "https://orcanotificationstorage.blob.core.windows.net/images/porttownsend.jpg";
            }
        }

        private static string GetModeratorEmailHtml(DateTime? timestamp, string location)
        {
            string timeString = GetPDTTimestring(timestamp);

            return $@"
                <body>
                <div class='card'>
                <h1>
                Orca Call Candidate
                </h1>
                <p>
                Dear moderator, a potential Southern Resident Killer Whale call was detected on {timeString} PDT at {location} location. 
                </p>
                <p>
                This is a request for your moderation to confirm whether the sound was produced by Southern Resident Killer Whale on the portal below.
                </p>
                <hr/>
                <h2>
                Orca Moderation Portal
                </h2>
                <p>
                Please click the link below to move to the portal. 
                </p>
                <a href='https://aifororcas.azurewebsites.net/' class='button-link'>
                Go to portal
                </a>
                </div>
                <footer>
                  <br>
                  <center>
                  In partnership with Microsoft AI 4 Earth, Orcasound and Orca Conservancy.
                  </center>
                </footer>
                </body>
            ";
        }

        private static string GetCSS()
        {
            return @"body {
                  font-family: 'Segoe UI', 'helvetica';
                  background-color: #F4F4F4;
                }

                .card {
                  background-color: white;
                  margin: 5%;
                  padding: 20px;
                }

                p {
                  color: dark-gray;
                }

                .button-link {
                  display: inline-block;
                  border: 0px;
                  background-color: #425AF4;
                  border-radius: 5px;
                  color: white;
                  font-size:20px;
                  padding:15px;
                  font-weight: bold;
                }

                footer {
                  text-align: center;
                  font-size:12px;
                }

                a {
                  text-decoration:none;
                  color: white;
                  font-weight: bold;
                }
            ";
        }

        private static string GetPDTTimestring(DateTime? timestamp)
        {
            var pacificTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
            return timestamp != null ? (TimeZoneInfo.ConvertTimeFromUtc(timestamp.Value, pacificTimeZone).ToShortDateString() + " " + TimeZoneInfo.ConvertTimeFromUtc(timestamp.Value, pacificTimeZone).ToLongTimeString()) : "unknown time";
        }
    }
}

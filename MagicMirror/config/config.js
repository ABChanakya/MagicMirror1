/* Standard MagicMirror profile
 *
 * Keeps the classic MagicMirror module layout while reusing the display and
 * network settings that work well on this hardware.
 */

let config = {
  address: "0.0.0.0",
  port: 8080,
  ipWhitelist: [],
  electronDisplay: 1,
  zoom: 1.3,

  language: "de",
  locale: "de-DE",
  timeFormat: 24,
  units: "metric",
  timezone: "Europe/Berlin",

  modules: [
    {
      module: "alert"
    },
    {
      module: "updatenotification",
      position: "top_bar"
    },
    {
      module: "clock",
      position: "top_left",
      config: {
        displaySeconds: false,
        timeFormat: 24,
        dateFormat: "dddd, D. MMMM"
      }
    },
    {
      module: "calendar",
      header: "Kalender",
      position: "top_left",
      config: {
        maximumEntries: 5,
        maximumNumberOfDays: 14,
        displaySymbol: true,
        calendars: [
          {
            fetchInterval: 7 * 24 * 60 * 60 * 1000,
            symbol: "calendar-check",
            url: "https://www.calendarlabs.com/ical-calendar/ics/69/Germany_Holidays.ics"
          }
        ]
      }
    },
    {
      module: "compliments",
      position: "lower_third",
      config: {
        compliments: {
          morning: [
            "Guten Morgen!",
            "Heute wird ein guter Tag."
          ],
          afternoon: [
            "Weiter so!",
            "Du machst das gut."
          ],
          evening: [
            "Schoenen Abend!",
            "Zeit zum Entspannen."
          ]
        }
      }
    },
    {
      module: "weather",
      position: "top_right",
      header: "Wetter jetzt",
      config: {
        weatherProvider: "openmeteo",
        type: "current",
        lat: 48.5442,
        lon: 12.1469
      }
    },
    {
      module: "weather",
      position: "top_right",
      header: "Vorhersage",
      config: {
        weatherProvider: "openmeteo",
        type: "forecast",
        lat: 48.5442,
        lon: 12.1469
      }
    },
    {
      module: "newsfeed",
      position: "bottom_bar",
      header: "Nachrichten",
      config: {
        feeds: [
          {
            title: "Sueddeutsche Zeitung",
            url: "https://rss.sueddeutsche.de/rss/Topthemen"
          }
        ],
        showSourceTitle: true,
        showPublishDate: true,
        broadcastNewsFeeds: true,
        broadcastNewsUpdates: true
      }
    }
  ]
};

if (typeof module !== "undefined") {
  module.exports = config;
}

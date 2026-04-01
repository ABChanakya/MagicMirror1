/* Legacy experimental profile
 *
 * Not the default setup. Use config.js for the standard profile and
 * config.child.js for the child-focused profile.
 *
 * Portrait 4K · German · 4 pages
 *
 * Pages:
 *   0 — Home        clock, weather, clothing, calendar, compliments
 *   1 — Fun         Pokemon, facts
 *   2 — Learn       On This Day, Quiz
 *   3 — Practical   newsfeed, calendar (full), NINA alerts
 */

const config = {
  address: "localhost",
  port: 8080,
  ipWhitelist: ["127.0.0.1", "::ffff:127.0.0.1", "::1"],

  language: "de",
  locale: "de-DE",
  timeFormat: 24,
  units: "metric",
  timezone: "Europe/Berlin",

  customCss: "css/custom.css",

  modules: [

    // ─── always visible ────────────────────────────────────────────────────

    { module: "alert" },

    {
      module: "MMM-CameraBridge",
      config: {
        port: 8081,
        facePageMap: {
          kind1: 1,
          kind2: 1,
          mama: 3,
          papa: 3,
        },
      },
    },

    {
      module: "MMM-Remote-Control",
      // No position — browser only, accessible at http://localhost:8080/remote.html
    },

    // Page indicator — bottom center, always visible
    {
      module: "MMM-page-indicator",
      position: "bottom_bar",
      config: {
        pages: 4,
      },
    },

    // Pages grouping module — no position needed
    {
      module: "MMM-pages",
      config: {
        modules: [
          // Page 0 — Home
          ["clock_home", "weather_current", "weather_forecast", "MMM-ClothingAdvice", "calendar_home", "compliments"],
          // Page 1 — Fun
          ["MMM-DailyPokemon", "MMM-Facts"],
          // Page 2 — Learn
          ["MMM-OnThisDay", "MMM-Quiz"],
          // Page 3 — Practical
          ["newsfeed", "calendar_practical", "MMM-NINA"],
        ],
        fixed: ["MMM-CameraBridge", "MMM-Remote-Control", "MMM-page-indicator", "alert"],
        animationTime: 400,
      },
    },

    // ─── Page 0 — Home ─────────────────────────────────────────────────────

    {
      module: "clock",
      classes: "clock_home",
      position: "top_center",
      config: {
        displaySeconds: false,
        timeFormat: 24,
        showDate: true,
        dateFormat: "dddd, D. MMMM",
      },
    },

    {
      module: "weather",
      classes: "weather_current",
      position: "middle_center",
      header: "Wetter jetzt",
      config: {
        weatherProvider: "openweathermap",
        type: "current",
        location: "Landshut",
        locationID: "2881742",
        apiKey: "d4bed3d835398ac481607eb26bcb8678",
        lang: "de",
        units: "metric",
        showWindDirection: true,
        showWindDirectionAsArrow: true,
        showHumidity: true,
        showFeelsLike: true,
        broadcastCurrentWeather: true,
      },
    },

    {
      module: "weather",
      classes: "weather_forecast",
      position: "middle_center",
      header: "5-Tage-Vorhersage",
      config: {
        weatherProvider: "openweathermap",
        type: "forecast",
        location: "Landshut",
        locationID: "2881742",
        apiKey: "d4bed3d835398ac481607eb26bcb8678",
        lang: "de",
        units: "metric",
        maxNumberOfDays: 5,
        showPrecipitationAmount: true,
      },
    },

    {
      module: "MMM-ClothingAdvice",
      classes: "MMM-ClothingAdvice",
      position: "middle_center",
      header: "Was anziehen?",
    },

    {
      module: "calendar",
      classes: "calendar_home",
      position: "lower_third",
      header: "Termine",
      config: {
        maximumEntries: 5,
        maximumNumberOfDays: 7,
        displaySymbol: true,
        calendars: [
          {
            symbol: "calendar-check",
            // Replace with your family iCal URL
            url: "https://www.calendarlabs.com/ical-calendar/ics/69/Germany_Holidays.ics",
          },
        ],
      },
    },

    {
      module: "compliments",
      classes: "compliments",
      position: "lower_third",
      config: {
        updateInterval: 30000,
        compliments: {
          morning: [
            "Guten Morgen! Starte super in den Tag!",
            "Guten Morgen! Du schaffst das heute!",
            "Ein neuer Tag voller Möglichkeiten!",
            "Morgen! Mach heute etwas Tolles!",
          ],
          afternoon: [
            "Weiter so!",
            "Du machst das großartig!",
            "Super, keep going!",
            "Toller Nachmittag!",
          ],
          evening: [
            "Schönen Abend!",
            "Zeit zum Entspannen.",
            "Du hast heute alles gegeben!",
            "Gute Erholung heute Abend!",
          ],
        },
      },
    },

    // ─── Page 1 — Fun ──────────────────────────────────────────────────────

    {
      module: "MMM-DailyPokemon",
      position: "middle_center",
      config: {
        language: "de",
        useGermanPokedex: true,
        // After receiving POKEMON_FLIP notification, module shows stats
        updateInterval: 86400000, // once per day
      },
    },

    {
      module: "MMM-Facts",
      position: "lower_third",
      config: {
        updateInterval: 60000,
      },
    },

    // ─── Page 2 — Learn ────────────────────────────────────────────────────

    {
      module: "MMM-OnThisDay",
      position: "upper_third",
      config: {
        language: null,  // null = inherit global "de"
        updateInterval: 3600000,
      },
    },

    {
      module: "MMM-Quiz",
      position: "middle_center",
      config: {
        // Answers selectable via gesture: hold 1/2/3/4 fingers
        // The module listens to QUIZ_ANSWER notifications from MMM-CameraBridge
        updateInterval: 60000,
      },
    },

    // ─── Page 3 — Practical ────────────────────────────────────────────────

    {
      module: "newsfeed",
      classes: "newsfeed",
      position: "upper_third",
      header: "Süddeutsche Zeitung",
      config: {
        feeds: [
          {
            title: "Süddeutsche Zeitung",
            url: "https://rss.sueddeutsche.de/rss/Topthemen",
          },
        ],
        showSourceTitle: true,
        showPublishDate: true,
        broadcastNewsFeeds: true,
      },
    },

    {
      module: "calendar",
      classes: "calendar_practical",
      position: "middle_center",
      header: "Kalender",
      config: {
        maximumEntries: 12,
        maximumNumberOfDays: 30,
        displaySymbol: true,
        calendars: [
          {
            symbol: "calendar-check",
            url: "https://www.calendarlabs.com/ical-calendar/ics/69/Germany_Holidays.ics",
          },
        ],
      },
    },

    {
      module: "MMM-NINA",
      position: "lower_third",
      header: "Warnungen",
      config: {
        // AGS (Amtlicher Gemeindeschlüssel) for Landshut
        ags: "09261000",
        updateInterval: 600000,
        maxAgeMinutes: 120,
      },
    },

  ],
};

if (typeof module !== "undefined") {
  module.exports = config;
}

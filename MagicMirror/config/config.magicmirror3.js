let config = {
  address: "0.0.0.0",
  port: 8081,
  ipWhitelist: [],
  electronDisplay: 1,
  zoom: 1.3,
  customCss: "css/custom.css",

  language: "de",
  locale: "de-DE",
  timeFormat: 24,
  units: "metric",
  timezone: "Europe/Berlin",

  modules: [
    // --- FIXED (visible on all pages) ---
    {
      module: "alert",
      classes: "fixed_page"
    },
    {
      module: "clock",
      position: "top_center",
      classes: "fixed_page portrait-clock",
      config: {
        displaySeconds: false,
        timeFormat: 24
      }
    },
    {
      module: "MMM-page-indicator",
      position: "bottom_center",
      classes: "fixed_page portrait-indicator",
      config: {
        pages: 4,
        activeBright: true,
        inactiveDimmed: true,
        inactiveHollow: true
      }
    },
    {
      module: "MMM-pages",
      classes: "fixed_page",
      config: {
        modules: [
          ["page1"],
          ["page2"],
          ["page3"],
          ["page4"]
        ],
        fixed: ["fixed_page"],
        animationTime: 800,
        homePage: 0,
        rotationDelay: 0 // disabled: navigation via gesture/keyboard only
      }
    },
    // Keyboard fallback (arrow keys / space to navigate while camera isn't running)
    {
      module: "MMM-Keypress",
      position: "bottom_right",
      classes: "fixed_page",
      hiddenOnStartup: true,
      config: {
        notifications: [
          { key: "Space", notification: "PAGE_INCREMENT" },
          { key: "Enter", notification: "PAGE_INCREMENT" },
          { key: "ArrowRight", notification: "PAGE_INCREMENT" },
          { key: "ArrowLeft", notification: "PAGE_DECREMENT" },
          { key: "Home", notification: "HOME_PAGE" }
        ]
      }
    },
    // Camera / gesture / face bridge (HTTP server on port 8082)
    {
      module: "MMM-CameraBridge",
      classes: "fixed_page",
      config: {
        port: 8082
      }
    },

    // --- PAGE 1: HOME ---
    {
      module: "weather",
      position: "upper_third",
      classes: "page1 portrait-card portrait-weather",
      header: "Wetter jetzt",
      config: {
        weatherProvider: "openweathermap",
        type: "current",
        location: "Landshut",
        locationID: "2881922",
        apiKey: "d4bed3d835398ac481607eb26bcb8678",
        units: "metric",
        showHumidity: "wind"
      }
    },
    {
      module: "MMM-ClothingAdvice",
      position: "middle_center",
      classes: "page1 portrait-card portrait-clothing",
      header: "Was soll ich anziehen?"
    },
    {
      module: "compliments",
      position: "lower_third",
      classes: "page1 portrait-card portrait-compliments",
      config: {
        compliments: {
          morning: [
            "Guten Morgen!",
            "Heute wartet ein Abenteuer auf dich!",
            "Du schaffst das!"
          ],
          afternoon: [
            "Weiter so!",
            "Zeit fuer die naechste Mission!",
            "Richtig stark!"
          ],
          evening: [
            "Schoener Abend!",
            "Heute war ein guter Tag.",
            "Jetzt ist Zeit zum Entspannen."
          ]
        }
      }
    },

    // --- PAGE 2: FUN (Kinder) ---
    {
      module: "MMM-DailyPokemon",
      position: "middle_center",
      classes: "page2 portrait-card portrait-pokemon",
      header: "Pokemon des Tages",
      config: {
        updateInterval: 86400000,
        minPoke: 1,
        maxPoke: 151,
        grayscale: false,
        showType: true,
        stats: true,
        language: "de",
        genera: true,
        gbaMode: true,
        nameSize: 30,
        flavorText: true
      }
    },
    {
      module: "MMM-Facts",
      position: "lower_third",
      classes: "page2 portrait-card portrait-facts",
      header: "Wusstest du schon?",
      config: {
        updateInterval: 10,
        fadeSpeed: 3,
        category: "deutsch",
        facts: {
          deutsch: [
            "Bienen muessen fuer ein Kilo Honig mehrere Millionen Blueten besuchen.",
            "Oktopusse haben drei Herzen.",
            "Der Eiffelturm kann im Sommer ein paar Zentimeter groesser werden.",
            "Koalas schlafen oft bis zu 20 Stunden am Tag.",
            "Ein Blitz ist heisser als die Oberflaeche der Sonne.",
            "Bananen sind Beeren, Erdbeeren aber nicht.",
            "Wombats machen wuerfelfoermigen Kot.",
            "Der erste Wecker konnte nur zu einer einzigen Uhrzeit klingeln.",
            "Es gibt mehr Sterne im Universum als Sandkoerner an vielen Straenden zusammen.",
            "Honig kann sehr lange haltbar bleiben.",
            "Pinguine koennen zwar nicht fliegen, aber sehr gut schwimmen.",
            "Manche Schildkroeten koennen ueber hundert Jahre alt werden.",
            "Der Mensch hat mehr Knochen in Haenden und Fuessen als im restlichen Koerper.",
            "Regenbogen entstehen, wenn Licht in Wassertropfen gebrochen wird.",
            "Der Mond entfernt sich jedes Jahr ein kleines Stueck von der Erde."
          ]
        }
      }
    },

    // --- PAGE 3: LEARN ---
    {
      module: "MMM-Quiz",
      position: "upper_third",
      classes: "page3 portrait-card portrait-quiz",
      header: "Quiz",
      config: {
        maxWidth: "100%",
        align: "center",
        questionsFile: "config/quiz.de.json",
        loadingText: "Quiz wird geladen ...",
        errorText: "Quizdaten konnten nicht geladen werden."
      }
    },
    {
      module: "MMM-OnThisDay",
      position: "lower_third",
      classes: "page3 portrait-card portrait-history",
      header: "Heute in der Geschichte",
      config: {
        wikipedia: true,
        maxItems: 4,
        maxWidth: "100%",
        textSize: "small"
      }
    },

    // --- PAGE 4: PRACTICAL ---
    {
      module: "calendar",
      position: "upper_third",
      classes: "page4 portrait-card portrait-calendar",
      header: "Kalender",
      config: {
        calendars: [
          {
            symbol: "calendar-check",
            url: "https://www.calendarlabs.com/ical-calendar/ics/69/Germany_Holidays.ics"
          }
        ]
      }
    },
    {
      module: "weather",
      position: "middle_center",
      classes: "page4 portrait-card portrait-forecast",
      header: "Vorhersage",
      config: {
        weatherProvider: "openweathermap",
        type: "forecast",
        location: "Landshut",
        locationID: "2881922",
        apiKey: "d4bed3d835398ac481607eb26bcb8678",
        units: "metric"
      }
    },
    {
      module: "MMM-NINA",
      position: "middle_center",
      classes: "page4 portrait-card portrait-nina",
      header: "Warnungen",
      config: {
        ags: "09274", // Landshut city
        degreeOfSeverity: 0,
        updateInterval: 5 * 60 * 1000
      }
    },
    {
      module: "newsfeed",
      position: "lower_third",
      classes: "page4 portrait-card portrait-news",
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

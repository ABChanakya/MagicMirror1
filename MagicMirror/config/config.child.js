/* Child-focused MagicMirror profile — 6 gesture-mapped pages
 *
 * Compact layout for Jetson Nano. Gesture mapping:
 *   Page 0 (Home):         Clock + Weather + Greeting
 *   Page 1 (fingers_1):    Weather + Clothing Advice
 *   Page 2 (fingers_2):    Pokémon Card (flip with 5 fingers)
 *   Page 3 (fingers_3):    Calendar + Facts
 *   Page 4 (fingers_4):    News Feed
 *   Page 5 (fingers_5):    History + Compliments
 *   Fist:                  Return to home (page 0)
 */

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
    {
      module: "alert",
      classes: "fixed_page"
    },
    {
      module: "MMM-CameraBridge",
      classes: "fixed_page",
      config: {
        port: 8082,
        fingerPageMap: {
          fingers_1: 1,
          fingers_2: 2,
          fingers_3: 3,
          fingers_4: 4,
          fist: 0,
        }
      }
    },
    {
      module: "clock",
      position: "top_center",
      classes: "fixed_page home-clock",
      config: {
        displaySeconds: false,
        timeFormat: 24
      }
    },
    {
      module: "MMM-SmartPresence",
      position: "fullscreen_above",
      classes: "fixed_page",
      config: {
        bridgeUrl: "ws://172.17.0.1:8084",
        greetingDuration: 8000,
        unknownName: "Gast",
        gestureMap: {
          fingers_1: { notification: "PAGE_SELECT", payload: 1 },
          fingers_2: { notification: "PAGE_SELECT", payload: 2 },
          fingers_3: { notification: "PAGE_SELECT", payload: 3 },
          fingers_4: { notification: "PAGE_SELECT", payload: 4 },
          fingers_5: { notification: "POKEMON_FLIP" },
          fist:      { notification: "PAGE_SELECT", payload: 0 },
        }
      }
    },
    {
      module: "MMM-page-indicator",
      position: "bottom_center",
      classes: "fixed_page",
      config: {
        pages: 6,
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
          ["home"],
          ["page1"],
          ["page2"],
          ["page3"],
          ["page4"],
          ["page5"]
        ],
        fixed: ["fixed_page"],
        animationTime: 600,
        homePage: 0,
        rotationDelay: 0
      }
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 0 (HOME) — Clock + Weather
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "weather",
      position: "middle_center",
      classes: "home portrait-weather",
      header: "Wetter jetzt",
      config: {
        weatherProvider: "openmeteo",
        type: "current",
        lat: 48.5442,
        lon: 12.1469,
        showHumidity: "wind"
      }
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 1 (fingers_1) — Weather + Clothing Advice
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "weather",
      position: "upper_third",
      classes: "page1 portrait-weather",
      header: "Wetter",
      config: {
        weatherProvider: "openmeteo",
        type: "current",
        lat: 48.5442,
        lon: 12.1469,
        showHumidity: "wind"
      }
    },
    {
      module: "MMM-ClothingAdvice",
      position: "lower_third",
      classes: "page1 portrait-clothing",
      header: "Was anziehen?"
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 2 (fingers_2) — Pokémon Card (flip with 5 fingers)
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "MMM-DailyPokemon",
      position: "middle_center",
      classes: "page2 portrait-pokemon",
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
        nameSize: 28,
        flavorText: false
      }
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 3 (fingers_3) — Calendar + Facts
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "calendar",
      position: "upper_third",
      classes: "page3 portrait-calendar",
      header: "Kalender",
      config: {
        calendars: [
          {
            symbol: "calendar-check",
            url: "https://www.calendarlabs.com/ical-calendar/ics/69/Germany_Holidays.ics"
          }
        ],
        maximumEntries: 4
      }
    },
    {
      module: "MMM-Facts",
      position: "lower_third",
      classes: "page3 portrait-facts",
      header: "Wusstest du schon?",
      config: {
        updateInterval: 15,
        fadeSpeed: 3,
        category: "deutsch",
        facts: {
          deutsch: [
            "Bienen müssen für ein Kilo Honig mehrere Millionen Blüten besuchen.",
            "Oktopusse haben drei Herzen.",
            "Der Eiffelturm kann im Sommer ein paar Zentimeter größer werden.",
            "Koalas schlafen oft bis zu 20 Stunden am Tag.",
            "Ein Blitz ist heißer als die Oberfläche der Sonne.",
            "Bananen sind Beeren, Erdbeeren aber nicht.",
            "Der erste Wecker konnte nur zu einer einzigen Uhrzeit klingeln.",
            "Es gibt mehr Sterne im Universum als Sandkörner an vielen Stränden.",
            "Pinguine können zwar nicht fliegen, aber sehr gut schwimmen.",
            "Der Mensch hat mehr Knochen in Händen und Füßen als im restlichen Körper.",
            "Regenbogen entstehen, wenn Licht in Wassertropfen gebrochen wird."
          ]
        }
      }
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 4 (fingers_4) — News Feed
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "newsfeed",
      position: "middle_center",
      classes: "page4 portrait-news",
      header: "Nachrichten",
      config: {
        feeds: [
          {
            title: "Süddeutsche Zeitung",
            url: "https://rss.sueddeutsche.de/rss/Topthemen"
          }
        ],
        showSourceTitle: true,
        showPublishDate: true,
        broadcastNewsFeeds: true,
        broadcastNewsUpdates: true,
        maxNewsItems: 5
      }
    },

    // ══════════════════════════════════════════════════════════════════════════
    // PAGE 5 (fingers_5) — History + Compliments
    // ══════════════════════════════════════════════════════════════════════════
    {
      module: "MMM-OnThisDay",
      position: "upper_third",
      classes: "page5 portrait-history",
      header: "Heute in der Geschichte",
      config: {
        wikipedia: true,
        maxItems: 3,
        maxWidth: "100%",
        textSize: "small"
      }
    },
    {
      module: "compliments",
      position: "lower_third",
      classes: "page5 portrait-compliments",
      header: "Motivation",
      config: {
        compliments: {
          morning: [
            "Guten Morgen! 🌅",
            "Heute wartet ein Abenteuer auf dich! 🚀",
            "Du schaffst das! 💪"
          ],
          afternoon: [
            "Weiter so! ✨",
            "Zeit für die nächste Mission! 🎯",
            "Richtig stark! ⭐"
          ],
          evening: [
            "Schöner Abend! 🌙",
            "Heute war ein guter Tag. 🎉",
            "Jetzt ist Zeit zum Entspannen. 😊"
          ]
        }
      }
    }
  ]
};

if (typeof module !== "undefined") {
  module.exports = config;
}

/**
 * MMM-ClothingAdvice
 *
 * Listens to the CURRENT_WEATHER notification broadcast by the default
 * weather module and shows child-friendly clothing icons + German advice text.
 *
 * No external API — works entirely from the weather data already on screen.
 *
 * Clothing icons live in modules/MMM-ClothingAdvice/icons/ as SVG files.
 * Full set: winter-coat, tshirt, shorts, umbrella, umbrella-closed, scarf,
 *   gloves, sunglasses, boots, sneakers, socks, cap, sun-hat, jeans, dress,
 *   snowflake, sun, rain-cloud, wind, thermometer, rainbow, fog, lightning, snowman
 */

Module.register("MMM-ClothingAdvice", {

  defaults: {},

  weatherData: null,

  start() {
    Log.info("MMM-ClothingAdvice: waiting for weather data...");
  },

  notificationReceived(notification, payload) {
    if (notification === "CURRENT_WEATHER_DATA") {
      this.weatherData = payload;
      this.updateDom(500);
    }
  },

  // Returns { icons: [...filenames...], advice: "..." } based on temp + conditions
  getClothingRecommendation(weather) {
    const temp  = weather.temperature;
    const type  = (weather.weatherType || "").toLowerCase();
    const rain  = type.includes("rain") || type.includes("drizzle") || (weather.rain && weather.rain > 0);
    const snow  = type.includes("snow");
    const sunny = type.includes("clear") || type.includes("sunny");
    const foggy = type.includes("fog") || type.includes("mist");
    const storm = type.includes("storm") || type.includes("thunder");
    const windy = (weather.windSpeed || 0) > 30;

    const icons = [];
    let advice = "";

    if (temp <= -10) {
      icons.push("winter-coat", "scarf", "gloves", "boots");
      advice = "Eisig kalt! Alles Warme anziehen – Jacke, Schal, Handschuhe und Stiefel.";
    } else if (temp <= -5) {
      icons.push("winter-coat", "scarf", "gloves");
      if (snow) icons.push("snowflake", "boots");
      advice = snow
        ? "Sehr kalt und Schnee! Warme Winterkleidung und Stiefel."
        : "Sehr kalt! Winterjacke, Schal und Handschuhe unbedingt anziehen.";
    } else if (temp <= 5) {
      icons.push("winter-coat", "scarf");
      if (snow) { icons.push("snowflake", "boots"); }
      advice = snow
        ? "Schnee draußen! Warme Kleidung und wasserfeste Stiefel."
        : "Kalt – Winterjacke und Schal nicht vergessen.";
    } else if (temp <= 10) {
      icons.push("winter-coat");
      if (rain) icons.push("umbrella");
      else if (windy) icons.push("scarf");
      advice = rain
        ? "Frisch und Regen – Jacke und Regenschirm einpacken."
        : "Frisch – eine warme Jacke ist heute ein Muss.";
    } else if (temp <= 15) {
      icons.push("winter-coat");
      if (rain) icons.push("umbrella");
      advice = rain
        ? "Mild, aber Regen – Jacke und Schirm mitnehmen."
        : "Angenehm kühl – eine Jacke ist sinnvoll.";
    } else if (temp <= 20) {
      icons.push("tshirt", "jeans");
      if (rain) icons.push("umbrella-closed");
      advice = rain
        ? "Mild mit Regen – leichte Jacke und Schirm bereithalten."
        : "Schönes Wetter – T-Shirt und Jeans passen gut.";
    } else if (temp <= 25) {
      icons.push("tshirt");
      if (sunny) icons.push("sunglasses", "cap");
      advice = sunny
        ? "Warm und sonnig – T-Shirt, Sonnenbrille und Mütze top!"
        : "Warm – ein T-Shirt reicht völlig.";
    } else {
      icons.push("tshirt", "shorts");
      if (sunny) icons.push("sunglasses", "sun-hat");
      advice = sunny
        ? "Heiß und sonnig! Leichte Kleidung, Sonnenhut und Brille."
        : "Sehr warm – T-Shirt und kurze Hose sind perfekt.";
    }

    // Extra conditions
    if (storm) {
      icons.push("lightning");
      advice += " Gewitter möglich – lieber drinnen bleiben!";
    }
    if (foggy) {
      icons.push("fog");
      advice += " Neblig – vorsichtig im Straßenverkehr.";
    }
    if (windy && !icons.includes("scarf") && temp < 18) {
      icons.push("wind");
      advice += " Windig – Jacke empfohlen.";
    }

    return { icons, advice };
  },

  getDom() {
    const wrapper = document.createElement("div");
    wrapper.className = "clothing-wrapper";

    if (!this.weatherData) {
      const loading = document.createElement("div");
      loading.className = "clothing-loading";
      loading.innerText = "Wetterdaten werden geladen...";
      wrapper.appendChild(loading);
      return wrapper;
    }

    const rec = this.getClothingRecommendation(this.weatherData);

    // Icon row
    const iconRow = document.createElement("div");
    iconRow.className = "clothing-icons";

    rec.icons.forEach(iconName => {
      const wrap = document.createElement("div");
      wrap.className = "clothing-icon-item";
      const img = document.createElement("img");
      img.src = this.file("icons/" + iconName + ".svg");
      img.alt = iconName;
      img.onerror = () => { img.style.display = "none"; }; // silently skip missing icons
      wrap.appendChild(img);
      iconRow.appendChild(wrap);
    });

    wrapper.appendChild(iconRow);

    // Advice text
    const text = document.createElement("div");
    text.className = "clothing-advice-text";
    text.innerText = rec.advice;
    wrapper.appendChild(text);

    return wrapper;
  },
});

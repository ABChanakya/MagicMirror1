/**
 * MMM-CameraBridge
 *
 * Receives gesture/face/presence events from the camera pipeline via a local
 * HTTP server and translates them into MagicMirror notifications.
 *
 * Event contract (POST /camera-event, JSON body):
 *   { "type": "gesture", "name": "swipe_left" }
 *   { "type": "gesture", "name": "swipe_right" }
 *   { "type": "gesture", "name": "swipe_up" }
 *   { "type": "gesture", "name": "swipe_down" }
 *   { "type": "gesture", "name": "fingers_1" }   <- quiz answer A
 *   { "type": "gesture", "name": "fingers_2" }   <- quiz answer B
 *   { "type": "gesture", "name": "fingers_3" }   <- quiz answer C
 *   { "type": "gesture", "name": "fingers_4" }   <- quiz answer D
 *   { "type": "face",    "profile": "kind1", "confidence": 0.92 }
 *   { "type": "face",    "profile": "unknown" }
 *   { "type": "presence","state": "present" }
 *   { "type": "presence","state": "away" }
 */

Module.register("MMM-CameraBridge", {

  defaults: {
    port: 8081,
    // Map face profile names to page indices (0-based)
    facePageMap: {
      kind1: 1,   // Fun page
      kind2: 1,
      mama:  3,   // Practical page
      papa:  3,
    },
    // Gestures that navigate pages globally
    pageNextGesture: "swipe_left",
    pagePrevGesture: "swipe_right",
    pageHomeGesture: "swipe_down",
    // Gesture to interact with the current page's main module
    interactGesture: "swipe_up",
    // Presence away -> dim display
    dimOnAway: true,
  },

  quizMode: false,

  start() {
    Log.info("MMM-CameraBridge: starting HTTP listener on port " + this.config.port);
    this.sendSocketNotification("START_SERVER", { port: this.config.port });
  },

  socketNotificationReceived(notification, payload) {
    if (notification === "CAMERA_EVENT") {
      this.handleCameraEvent(payload);
    }
  },

  notificationReceived(notification) {
    if (notification === "QUIZ_QUESTION_START") {
      this.quizMode = true;
    }
    if (notification === "QUIZ_QUESTION_END" || notification === "QUIZ_STOP") {
      this.quizMode = false;
    }
  },

  handleCameraEvent(event) {
    Log.info("MMM-CameraBridge: event", JSON.stringify(event));

    switch (event.type) {
      case "gesture":
        this.handleGesture(event.name);
        break;

      case "face":
        if (event.profile && event.profile !== "unknown") {
          const page = this.config.facePageMap[event.profile];
          if (page !== undefined) {
            this.sendNotification("PAGE_SELECT", page);
          }
        }
        break;

      case "presence":
        if (this.config.dimOnAway) {
          this.sendNotification("USER_PRESENCE", event.state === "present");
        }
        break;
    }
  },

  handleGesture(name) {
    // In quiz mode, finger gestures map to answers
    if (this.quizMode) {
      const answerMap = { fingers_1: 0, fingers_2: 1, fingers_3: 2, fingers_4: 3 };
      if (answerMap[name] !== undefined) {
        this.sendNotification("QUIZ_ANSWER", answerMap[name]);
        return;
      }
    }

    // Swipe up = interact with current page (flip Pokemon, etc.)
    if (name === this.config.interactGesture) {
      this.sendNotification("POKEMON_FLIP");
      return;
    }

    // Global page navigation
    switch (name) {
      case this.config.pageNextGesture:
        this.sendNotification("PAGE_INCREMENT");
        break;
      case this.config.pagePrevGesture:
        this.sendNotification("PAGE_DECREMENT");
        break;
      case this.config.pageHomeGesture:
        this.sendNotification("PAGE_SELECT", 0);
        break;
    }
  },

  getDom() {
    return document.createElement("div");
  },
});

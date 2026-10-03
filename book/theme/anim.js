// Controls for the book's animations: pause, speed, and a button for each step.
//
// Every animation is a short looping video. A reader who needs more time can
// slow it down, pause it, jump back to a step, or have it stop at the end of
// each step and wait. A reader whose system asks for reduced motion gets the
// first frame and a play button.
(function () {
  "use strict";

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function button(label, title, onClick) {
    var b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    b.addEventListener("click", onClick);
    return b;
  }

  function setup(video) {
    var chapters = [];
    try { chapters = JSON.parse(video.dataset.chapters || "[]"); } catch (e) { chapters = []; }

    var bar = document.createElement("div");
    bar.className = "motion-bar";

    var play = button("", "Play or pause (or click the animation)", toggle);
    play.className = "motion-play";
    bar.appendChild(play);

    var speeds = document.createElement("span");
    speeds.className = "motion-group";
    [0.5, 0.75, 1].forEach(function (rate) {
      var b = button(rate + "×", "Play at " + rate + " speed", function () {
        video.playbackRate = rate;
        mark(speeds, b);
      });
      if (rate === 1) { b.setAttribute("aria-pressed", "true"); }
      speeds.appendChild(b);
    });
    bar.appendChild(speeds);

    var steps = document.createElement("span");
    steps.className = "motion-group motion-steps";
    var stepButtons = chapters.map(function (c, i) {
      var b = button((i + 1) + ". " + c[1], "Jump to this step", function () {
        video.currentTime = c[0] + 0.01;
        video.play();
      });
      steps.appendChild(b);
      return b;
    });
    if (chapters.length) { bar.appendChild(steps); }

    var stepLabel = document.createElement("label");
    stepLabel.className = "motion-stepwise";
    var stepwise = document.createElement("input");
    stepwise.type = "checkbox";
    stepLabel.appendChild(stepwise);
    stepLabel.appendChild(document.createTextNode(" pause after each step"));
    if (chapters.length > 1) { bar.appendChild(stepLabel); }

    video.insertAdjacentElement("afterend", bar);

    function mark(group, chosen) {
      Array.prototype.forEach.call(group.children, function (b) {
        b.setAttribute("aria-pressed", b === chosen ? "true" : "false");
      });
    }

    function toggle() {
      if (video.paused) { video.play(); } else { video.pause(); }
    }

    function current(t) {
      var k = 0;
      chapters.forEach(function (c, i) { if (t >= c[0]) { k = i; } });
      return k;
    }

    var last = 0;
    video.addEventListener("timeupdate", function () {
      var t = video.currentTime;
      var k = current(t);
      stepButtons.forEach(function (b, i) { b.setAttribute("aria-current", i === k ? "step" : "false"); });
      // Stop where the next step begins, so the reader decides when to go on.
      if (stepwise.checked && !video.paused && t > last && k !== current(last)) {
        video.pause();
        video.currentTime = chapters[k][0];
      }
      last = t;
    });
    video.addEventListener("play", function () { play.textContent = "❚❚ Pause"; });
    video.addEventListener("pause", function () { play.textContent = "▶ Play"; });
    video.addEventListener("click", toggle);

    if (reduce) {
      video.removeAttribute("autoplay");
      video.pause();
    }
    play.textContent = video.paused || reduce ? "▶ Play" : "❚❚ Pause";
  }

  document.querySelectorAll("video.motion").forEach(setup);
})();

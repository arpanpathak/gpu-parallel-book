// A click or a tap opens a diagram full screen, so its labels can be read on a
// phone. On an upright phone a wide diagram is turned a quarter to use the long
// side of the screen. Another tap, or Escape, closes it.
(function () {
    "use strict";

    function open(diagram) {
        var view = document.createElement("div");
        var picture = document.createElement("img");

        view.className = "figure-zoom";
        picture.src = diagram.src;
        picture.alt = diagram.alt;

        var upright = window.innerHeight > window.innerWidth;
        var wide = diagram.naturalWidth > diagram.naturalHeight * 1.2;

        if (upright && wide) {
            view.classList.add("turned");
        }

        function close() {
            view.remove();
            document.removeEventListener("keydown", onKey);
        }

        function onKey(event) {
            if (event.key === "Escape") {
                close();
            }
        }

        view.appendChild(picture);
        view.addEventListener("click", close);
        document.addEventListener("keydown", onKey);
        document.body.appendChild(view);
    }

    document.querySelectorAll('figure > img[src$=".svg"]').forEach(function (diagram) {
        diagram.style.cursor = "zoom-in";
        diagram.title = "Open at full size";
        diagram.addEventListener("click", function () { open(diagram); });
    });
})();

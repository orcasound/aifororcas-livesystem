(function () {
    var tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    var tileAttribution = 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>';
    let marker = {};
    // Global export
    window.deliveryMap = {
        showOrUpdate: function (elementId, markers) {
            var elem = document.getElementById(elementId);
            if (!elem) {
                throw new Error('No element with ID ' + elementId);
            }

            // Initialize map if needed
            if (!elem.map) {
                elem.map = L.map(elementId, { attributionControl: false }).setView([48.360235, -122.747209], 7); // Set view in the salish sea
                L.tileLayer(tileUrl, { attribution: tileAttribution }).addTo(elem.map);
            }
        },
        focusOnLocation: function(lat, lng, name) {
            const map = document.querySelector(".track-order-map [id^='map']").map;

            // Clear existing marker
            if (marker !== undefined) {
                map.removeLayer(marker);
            }
            const hydrophoneIcon = L.icon({
                iconUrl: 'images/hydrophone-blue.png',

                iconSize: [40, 40], // size of the icon
                popupAnchor: [0, -20] // point from which the popup should open relative to the iconAnchor
            })
            // Add a marker to show current selected location
            marker = L.marker([lat, lng], { icon: hydrophoneIcon }).addTo(map);
            marker.bindPopup(name).openPopup();
            map.setView([lat, lng], 10);
        },
        unFocus: function () {
            const map = document.querySelector(".track-order-map [id^='map']").map;

            // Clear existing marker
            if (marker !== undefined) {
                map.removeLayer(marker);
            }
        }
    };

    function animateMarkerMove(marker, coords, durationMs) {
        if (marker.existingAnimation) {
            cancelAnimationFrame(marker.existingAnimation.callbackHandle);
        }

        marker.existingAnimation = {
            startTime: new Date(),
            durationMs: durationMs,
            startCoords: { x: marker.getLatLng().lng, y: marker.getLatLng().lat },
            endCoords: coords,
            callbackHandle: window.requestAnimationFrame(() => animateMarkerMoveFrame(marker))
        };
    }

    function animateMarkerMoveFrame(marker) {
        var anim = marker.existingAnimation;
        var proportionCompleted = (new Date().valueOf() - anim.startTime.valueOf()) / anim.durationMs;
        var coordsNow = {
            x: anim.startCoords.x + (anim.endCoords.x - anim.startCoords.x) * proportionCompleted,
            y: anim.startCoords.y + (anim.endCoords.y - anim.startCoords.y) * proportionCompleted
        };

        marker.setLatLng([coordsNow.y, coordsNow.x]);

        if (proportionCompleted < 1) {
            marker.existingAnimation.callbackHandle = window.requestAnimationFrame(
                () => animateMarkerMoveFrame(marker));
        }
    }
})();

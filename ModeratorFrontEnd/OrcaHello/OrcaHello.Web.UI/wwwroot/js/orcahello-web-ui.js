/* Modal Map Functionality */

function LoadSmallBingMap(id, latitude, longitude) {
    var map = new Microsoft.Maps.Map(document.getElementById('small-bing-map-id-' + id), {
        center: new Microsoft.Maps.Location(latitude, longitude),
        mapTypeId: Microsoft.Maps.MapTypeId.aerial,
        zoom: 12
    });
    CreateScaledPushpin(map.getCenter(), 'img/hydrophone.png', .2, function (pin) {
        map.entities.push(pin);
    });
}

function CreateScaledPushpin(location, imgUrl, scale, callback) {
    var img = new Image();
    img.onload = function () {
        var c = document.createElement('canvas');
        c.width = img.width * scale;
        c.height = img.height * scale;

        var context = c.getContext('2d');

        //Draw scaled image
        context.drawImage(img, 0, 0, c.width, c.height);

        var pin = new Microsoft.Maps.Pushpin(location, {
            //Generate a base64 image URL from the canvas.
            icon: c.toDataURL(),

            //Anchor based on the center of the image.
            anchor: new Microsoft.Maps.Point(c.width / 2, c.height / 2)
        });

        if (callback) {
            callback(pin);
        }
    };

    img.src = imgUrl;
}

/* Spectrogram functionality */

function getBoundingClientRect(element) {
    // Get the bounding client rectangle of the element
    return element.getBoundingClientRect();
}

function repositionHowl(soundId, position) {
    const howl = getHowl(soundId);
    howl.seek(position);
}

function clearAllHowls() {
    Howler.stop();
    Howler.unload();
}
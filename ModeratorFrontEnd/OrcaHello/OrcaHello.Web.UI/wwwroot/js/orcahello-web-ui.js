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

function playAudio(id) {
    var audio = document.getElementById('audio_' + id);
    audio.play();
}

function pauseAudio(id) {
    var audio = document.getElementById('audio_' + id);
    audio.pause();
}

function stopAudio(id) {

    var audio = document.getElementById('audio_' + id);
    audio.pause();
    audio.currentTime = 0;
    // Reset the line position when stopped
    document.getElementById('line_' + id).style.left = "0px";
}

function formatTime(seconds) {
    var minutes = Math.floor(seconds / 60);
    var seconds = Math.floor(seconds % 60);
    return minutes + ":" + (seconds < 10 ? "0" + seconds : seconds);
}

function updateTimeAndLine(audio) {

    var audio = document.getElementById(audio.id);

    var id = audio.id.slice(6);
    var currentTime = audio.currentTime;

    // Update the time display
    document.getElementById('time_' + id).innerHTML = formatTime(currentTime) + " / " + formatTime(audio.duration);
    // Calculate the line position based on the audio progress and the image width
    var imageWidth = document.getElementById('spectrogram_' + id).width;
    var linePosition = imageWidth * currentTime / audio.duration;
    // Use requestAnimationFrame to smooth the line movement
    requestAnimationFrame(function () {
        // Move the line to the calculated position
        document.getElementById('line_' + id).style.left = linePosition + "px";
    });
}

function seekAudio(event, id) {

    // Get the audio element by id
    var audio = document.getElementById('audio_' + id);

    // Get the mouse position relative to the image
    var mouseX = event.clientX - document.documentElement.scrollLeft;

    // Get the image element
    var image = document.getElementById('spectrogram_' + id);
    // Loop through all the parent elements and subtract their offsetLeft values
    while (image.offsetParent) {
        mouseX -= image.offsetLeft;
        image = image.offsetParent;
    }

    var image = document.getElementById('spectrogram_' + id);
    // Get the image width
    var imageWidth = image.width;
    // Calculate the audio time based on the mouse position and the image width
    var time = Number((mouseX * audio.duration / imageWidth).toFixed(0));

    // This is a recommended hack for setting currentTime
    audio.pause();
    audio.src = audio.src;
    audio.currentTime = time;

    updateTimeAndLine(audio);
    audio.play();
}

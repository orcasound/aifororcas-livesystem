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

function StartGridAudioPlayback(url, dotnetHelper) {
    var sound = new Howl({ src: [url], html5: true });

    var completionEvent = new CustomEvent("completionEvent", {
        detail: {
            Url: url
        }
    });

    sound.on("end", function () { dotnetHelper.invokeMethodAsync("OnJSCustomEvent", completionEvent) });
    sound.play();
}

function StopGridAudioPlayback() {
    Howler.stop();
    Howler.unload();
    console.log("Unloaded");
}

/* Single Page Spectrogram functionality */

function SetupPlayback(audioFileUrl, id) {

    var sound = new Howl({ src: [audioFileUrl], html5: false });

    // Get references to the HTML elements
    var play = document.getElementById('play_' + id);
    var pause = document.getElementById('pause_' + id);
    var stop = document.getElementById('stop_' + id);
    var volume = document.getElementById('volume_' + id);
    var progress = document.getElementById('progress_' + id);
    var timer = document.getElementById('timer_' + id);
    var image = document.getElementById('image_' + id);

    // Add event listeners to the buttons
    play.addEventListener('click', function () {
        sound.play();
    });

    pause.addEventListener('click', function () {
        sound.pause();
    });

    stop.addEventListener('click', function () {
        sound.stop();
    });

    volume.addEventListener('input', function () {
        sound.volume(volume.value);
    });

    image.addEventListener('click', function (e) {

        // Get the x coordinate of the mouse pointer relative to the image element
        var x = e.offsetX;
        // Get the width of the image element
        var width = image.offsetWidth;
        // Calculate the percentage value of the click position
        var percentage = (x / width) * 100;

        // Set the new position of the progress element
        progress.style.left = percentage + '%';

        // Calculate the new position of the sound in seconds
        var duration = sound.duration();

        var position = (percentage / 100) * duration;

        // Set the new position of the sound
        sound.seek(position);
    });

    // Add event listeners to the Howl object
    sound.on('play', function () {
        // Update the progress and timer every 100ms
        var interval = setInterval(function () {
            var seek = sound.seek() || 0; // Get the current position in seconds
            var duration = sound.duration(); // Get the total duration in seconds
            var percentage = (seek / duration) * 100; // Calculate the percentage of progress

            var minutes = Math.floor(seek / 60); // Calculate the minutes part of the timer
            var seconds = Math.floor(seek % 60); // Calculate the seconds part of the timer

            // Format the timer as mm:ss
            if (minutes < 10) minutes = '0' + minutes;
            if (seconds < 10) seconds = '0' + seconds;

            var durationMinutes = Math.floor(duration / 60); // Calculate the minutes part of the duration
            var durationSeconds = Math.floor(duration % 60); // Calculate the seconds part of the duration

            // Format the duration as mm:ss
            if (durationMinutes < 10) durationMinutes = '0' + durationMinutes; // Add a leading zero if needed
            if (durationSeconds < 10) durationSeconds = '0' + durationSeconds; // Add a leading zero if needed

            var timerText = minutes + ':' + seconds + ' / ' + durationMinutes + ':' + durationSeconds;

            // Update the progress and timer elements
            progress.style.left = percentage + '%';
            timer.textContent = timerText;
        }, 100);
    });

    sound.on('end', function () {
        // Clear the interval and reset the player when the audio has finished playing
        clearInterval(interval);
        progress.style.left = '0%';
        timer.textContent = '00:00 / 00:00';
    });
}

/* SmallSpectrogramPlayerComponent setup and execution */

// Declare a global variable to store the current sound
var currentSound = null;

// Declare a global variable to store the current position of the sound
var currentPosition = 0;

var currentInterval = null;

// Define a function to play a sound from the passed source
function StartSmallSpectrogramPlayback(id, audioFileUrl) {

    if (currentSound && currentSound._src == audioFileUrl) {
        currentSound.play();
    }
    else {
        // Stop the current sound if it exists and is not the referenced sound
        if (currentSound && currentSound._src !== audioFileUrl)
            currentSound.stop();

        // Create a new sound object from the source URL
        var sound = new Howl({
            src: [audioFileUrl],
            html5: true
        });

        var progress = document.getElementById('spectrogram-progress-indicator-' + id);
        var timer = document.getElementById('spectrogram-progress-timer-' + id);

        // Add event listeners to the Howl object
        sound.on('play', function () {
            // Update the progress and timer every 100ms
            currentInterval = setInterval(function () {

                console.log(currentPosition);

                currentPosition = sound.seek() || 0; // Get the current position in seconds
                var duration = sound.duration(); // Get the total duration in seconds
                var percentage = (currentPosition / duration) * 100; // Calculate the percentage of progress

                var minutes = Math.floor(currentPosition / 60); // Calculate the minutes part of the timer
                var seconds = Math.floor(currentPosition % 60); // Calculate the seconds part of the timer

                // Format the timer as mm:ss
                if (minutes < 10) minutes = '0' + minutes;
                if (seconds < 10) seconds = '0' + seconds;

                var durationMinutes = Math.floor(duration / 60); // Calculate the minutes part of the duration
                var durationSeconds = Math.floor(duration % 60); // Calculate the seconds part of the duration

                // Format the duration as mm:ss
                if (durationMinutes < 10) durationMinutes = '0' + durationMinutes; // Add a leading zero if needed
                if (durationSeconds < 10) durationSeconds = '0' + durationSeconds; // Add a leading zero if needed

                var timerText = minutes + ':' + seconds + ' / ' + durationMinutes + ':' + durationSeconds;

                // Update the progress and timer elements
                progress.style.left = percentage + '%';
                timer.textContent = timerText;
            }, 100);
        });

        sound.on('stop', function () {
            clearInterval(currentInterval);
            currentSound = null;
            currentPosition = 0;
        });

        // Play the sound
        sound.play();

        // Assign the sound to the current sound variable
        currentSound = sound;
    }
}


//sound = new Howl({ src: [audioFileUrl], html5: true });

//var progress = document.getElementById('spectrogram-progress-indicator-' + id);
//var timer = document.getElementById('spectrogram-progress-timer-' + id);

//// Add event listeners to the Howl object
//sound.on('play', function () {
//    // Update the progress and timer every 100ms
//    var interval = setInterval(function () {
//        var seek = sound.seek() || 0; // Get the current position in seconds
//        var duration = sound.duration(); // Get the total duration in seconds
//        var percentage = (seek / duration) * 100; // Calculate the percentage of progress

//        var minutes = Math.floor(seek / 60); // Calculate the minutes part of the timer
//        var seconds = Math.floor(seek % 60); // Calculate the seconds part of the timer

//        // Format the timer as mm:ss
//        if (minutes < 10) minutes = '0' + minutes;
//        if (seconds < 10) seconds = '0' + seconds;

//        var durationMinutes = Math.floor(duration / 60); // Calculate the minutes part of the duration
//        var durationSeconds = Math.floor(duration % 60); // Calculate the seconds part of the duration

//        // Format the duration as mm:ss
//        if (durationMinutes < 10) durationMinutes = '0' + durationMinutes; // Add a leading zero if needed
//        if (durationSeconds < 10) durationSeconds = '0' + durationSeconds; // Add a leading zero if needed

//        var timerText = minutes + ':' + seconds + ' / ' + durationMinutes + ':' + durationSeconds;

//        // Update the progress and timer elements
//        progress.style.left = percentage + '%';
//        timer.textContent = timerText;
//    }, 100);
//});

//sound.play();


function PauseSmallSpectrogramPlayback() {
    // Pause the current sound if it exists
    if (currentSound) {
        currentSound.pause();
    }
}

function StopSmallSpectrogramPlayback() {
    // Stop the current sound if it exists
    if (currentSound) {
        currentSound.stop();
    }
    // Set the current sound variable to null
    currentInterval = null;
    currentSound = null;
    currentPosition = 0;
}

// TODO: Remove when done with Player.razor

//var sound = new Howl({
//    src: ["https://livemlaudiospecstorage.blob.core.windows.net/audiowavs/rpi_orcasound_lab_2023_08_17_17_28_52_PDT.wav"], html5: true
//});
//// Get references to the HTML elements
//var play = document.getElementById('play');
//var pause = document.getElementById('pause');
//var stop = document.getElementById('stop');
//var mute = document.getElementById('mute');
//var volume = document.getElementById('volume');
//var progress = document.getElementById('progress');
//var timer = document.getElementById('timer');
//var image = document.getElementById('image'); // Get the image element

//// Add event listeners to the buttons
//play.addEventListener('click', function () {
//    sound.play();
//});

//pause.addEventListener('click', function () {
//    sound.pause();
//});

//stop.addEventListener('click', function () {
//    sound.stop();
//});

//mute.addEventListener('click', function () {
//    sound.mute(!sound.mute());
//});

//volume.addEventListener('input', function () {
//    sound.volume(volume.value);
//});

//// Add event listeners to the Howl object
//sound.on('play', function () {
//    // Update the progress and timer every 100ms
//    var interval = setInterval(function () {
//        var seek = sound.seek() || 0; // Get the current position in seconds
//        var duration = sound.duration(); // Get the total duration in seconds
//        var percentage = (seek / duration) * 100; // Calculate the percentage of progress
//        var minutes = Math.floor(seek / 60); // Calculate the minutes part of the timer
//        var seconds = Math.floor(seek % 60); // Calculate the seconds part of the timer
//        // Format the timer as mm:ss
//        if (minutes < 10) minutes = '0' + minutes;
//        if (seconds < 10) seconds = '0' + seconds;
//        // Format the duration as mm:ss
//        var durationMinutes = Math.floor(duration / 60); // Calculate the minutes part of the duration
//        var durationSeconds = Math.floor(duration % 60); // Calculate the seconds part of the duration
//        if (durationMinutes < 10) durationMinutes = '0' + durationMinutes; // Add a leading zero if needed
//        if (durationSeconds < 10) durationSeconds = '0' + durationSeconds; // Add a leading zero if needed
//        var timerText = minutes + ':' + seconds + ' / ' + durationMinutes + ':' + durationSeconds;

//        // Update the progress and timer elements
//        progress.style.left = percentage + '%';
//        timer.textContent = timerText;
//    }, 100);
//});

//sound.on('end', function () {
//    // Clear the interval and reset the player when the audio has finished playing
//    clearInterval(interval);
//    progress.style.left = '0%';
//    timer.textContent = '00:00 / 00:00';
//});

//// Add a click event listener to the image element
//image.addEventListener('click', function (e) {
//    // Get the x coordinate of the mouse pointer relative to the image element
//    var x = e.offsetX;
//    // Get the width of the image element
//    var width = image.offsetWidth;
//    // Calculate the percentage value of the click position
//    var percentage = (x / width) * 100;
//    // Calculate the new position of the sound in seconds
//    var duration = sound.duration();
//    var position = (percentage / 100) * duration;
//    // Set the new position of the sound
//    sound.seek(position);
//    alert(sound.seek());
//    // Set the new position of the progress element
//    progress.style.left = percentage + '%';
//});

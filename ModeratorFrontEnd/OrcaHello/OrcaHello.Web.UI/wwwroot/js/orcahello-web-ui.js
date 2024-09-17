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

window.addAudioEventListenersByObject = (audioElement) => {

    return new Promise((resolve, reject) => {

        if (!audioElement) {
            reject('Audio element is not loaded yet.');
            return;
        }

        const spectrogram = document.getElementById('spectrogram');
        const playbackLine = document.getElementById('playbackLine');

        audioElement.onloadedmetadata = () => {

            audioElement.ontimeupdate = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('timeupdate', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onseeking = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('seeking', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onseeked = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('seeked', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onpause = () => {
                // Pause the movement of the line
            };

            audioElement.onplay = () => {
                // Resume the movement of the line
            };

            audioElement.onended = () => {
                playbackLine.style.left = '0px';
            };

            resolve();
        };
    });
};

window.addAudioEventStopper = (audioId) => {

    var audioElement = document.getElementById('audio_' + audioId);
    audioElement.addEventListener('play', function () {

        var allAudioElements = document.getElementsByTagName('audio')
        
        // When an audio element is played, pause all other audio elements
        for (var j = 0; j < allAudioElements.length; j++) {
            if (allAudioElements[j].id != audioElement.id) {  // Don't pause the audio element that is being played
                allAudioElements[j].pause();
                allAudioElements[j].currentTime = 0;
            }
        }
    });
};

window.addAudioEventListenersById = (audioId) => {

    return new Promise((resolve, reject) => {

        var audioElement = document.getElementById('audio_' + audioId);

        if (!audioElement) {
            reject('Audio element is not loaded yet.');
            return;
        }

        const spectrogram = document.getElementById('spectrogram_' + audioId);
        const playbackLine = document.getElementById('playback_line_' + audioId);

        audioElement.onloadedmetadata = () => {

            audioElement.ontimeupdate = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('timeupdate', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onseeking = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('seeking', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onseeked = () => {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            };

            audioElement.addEventListener('seeked', function () {
                const progress = audioElement.currentTime / audioElement.duration;
                playbackLine.style.left = `${progress * spectrogram.clientWidth}px`;
            });

            audioElement.onpause = () => {
                // Pause the movement of the line
            };

            audioElement.onended = () => {
                playbackLine.style.left = '0px';
            };

            audioElement.addEventListener('play', function () {

                var allAudioElements = document.getElementsByTagName('audio');

                // When an audio element is played, pause all other audio elements
                for (var j = 0; j < allAudioElements.length; j++) {
                    if (allAudioElements[j].id != audioElement.id) {  // Don't pause the audio element that is being played
                        allAudioElements[j].pause();
                        allAudioElements[j].currentTime = 0;
                    }
                }
            });

            resolve();
        };
    });
};

window.stopAllAudio = () => {
    var allAudioElements = document.getElementsByTagName('audio');
    for (var j = 0; j < allAudioElements.length; j++) {
        allAudioElements[j].pause();
        allAudioElements[j].currentTime = 0;
    }
};
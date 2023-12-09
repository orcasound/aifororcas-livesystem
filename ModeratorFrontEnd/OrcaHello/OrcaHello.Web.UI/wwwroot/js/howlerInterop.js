window.howlerInterop = {
    sound: null,
    progressLine: null,
    updateRaf: null,

    playSound: function (soundFile) {
        if (!this.sound) {
            this.sound = new Howl({
                src: [soundFile],
                onplay: () => {
                    this.progressLine = document.getElementById('progressLine');
                    this.progressLine.style.display = 'block';
                    this.startProgress();
                },
                onstop: this.resetProgress,
                onpause: () => {
                    // Don't hide the progress line on pause
                }
            });
        }
        this.sound.play();
    },

    pauseSound: function () {
        if (this.sound && this.sound.playing()) {
            this.sound.pause();
        }
    },

    stopSound: function () {
        if (this.sound) {
            this.sound.stop();
        }
    },

    startProgress: function () {
        const updateProgress = () => {
            const seek = this.sound.seek() || 0;
            const duration = this.sound.duration();
            this.progressLine.style.left = ((seek / duration) * 100) + '%';
            this.updateRaf = requestAnimationFrame(updateProgress);
        };
        this.updateRaf = requestAnimationFrame(updateProgress);
    },

    resetProgress: function () {
        if (this.updateRaf) {
            cancelAnimationFrame(this.updateRaf);
        }
        this.progressLine.style.left = '0%';
        this.progressLine.style.display = 'none';
    }
};

//window.getOffsetWidth = function (element) {
//    return element.offsetWidth;
//};

//window.getBoundingClientRect = function (element) {
//    return element.getBoundingClientRect();
//};



//window.getOffsetWidth = function (element) {
//    return element.offsetWidth;
//};

//window.getBoundingClientRect = function (element) {
//    return element.getBoundingClientRect();
//};

//window.howlerInterop = {
//    sound: null,
//    progressLine: null,
//    updateRaf: null,

//    playSound: function (soundFile) {
//        if (!this.sound) {
//            this.sound = new Howl({
//                src: [soundFile],
//                onplay: () => {
//                    this.progressLine = document.getElementById('progressLine');
//                    this.progressLine.style.display = 'block';
//                    this.startProgress();
//                },
//                onstop: this.resetProgress,
//                onpause: () => {
//                    // Don't hide the progress line on pause
//                }
//            });
//        }
//        this.sound.play();
//    },

//    pauseSound: function () {
//        if (this.sound && this.sound.playing()) {
//            this.sound.pause();
//        }
//    },

//    stopSound: function () {
//        if (this.sound) {
//            this.sound.stop();
//        }
//    },

//    startProgress: function () {
//        const updateProgress = () => {
//            const seek = this.sound.seek() || 0;
//            const duration = this.sound.duration();
//            this.progressLine.style.left = ((seek / duration) * 100) + '%';
//            // this.updateTimeDisplay(seek, duration);
//            this.updateRaf = requestAnimationFrame(updateProgress);
//        };
//        this.updateRaf = requestAnimationFrame(updateProgress);
//    },

//    resetProgress: function () {
//        if (this.updateRaf) {
//            cancelAnimationFrame(this.updateRaf);
//        }
//        this.progressLine.style.left = '0%';
//        this.progressLine.style.display = 'none';
//    },

//    seekAudio: function (clickPosition, imageWidth) {
//        var duration = this.sound.duration();
//        var seekPosition = (clickPosition / imageWidth) * duration;
//        this.sound.seek(seekPosition);
//    },

    //updateTimeDisplay: function (seek, duration) {
    //    const formatTime = (secs) => {
    //        const minutes = Math.floor(secs / 60) || 0;
    //        const seconds = Math.floor(secs - minutes * 60) || 0;
    //        return minutes + ':' + (seconds < 10 ? '0' : '') + seconds;
    //    };

    //    const currentTime = formatTime(seek);
    //    const totalTime = formatTime(duration);
    //    document.getElementById('timeDisplay').textContent = currentTime + ' / ' + totalTime;
    //}
/*};*/

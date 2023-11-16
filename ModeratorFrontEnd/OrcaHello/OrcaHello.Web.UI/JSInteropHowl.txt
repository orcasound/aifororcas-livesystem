const howlInstances = {};

window.howl = {
    play: function (dotnetReference, options) {
        // Create new Howl instance
        const howl = new Howl({
            src: options.sources,
            format: options.formats,
            html5: options.html5,
            loop: options.loop,
            volume: options.volume,

            onplay: async function (id) {
                let duration = howl.duration(id);
                if (duration === Infinity || isNaN(duration)) {
                    duration = null;
                }
                await dotnetReference.invokeMethodAsync('OnPlayCallback', id, duration);
            },
            onstop: async function (id) {
                await dotnetReference.invokeMethodAsync('OnStopCallback', id);
            },
            onpause: async function (id) {
                await dotnetReference.invokeMethodAsync('OnPauseCallback', id);
            },
            onrate: async function (id) {
                const currentRate = howl.rate();
                await dotnetReference.invokeMethodAsync('OnRateCallback', id, currentRate);
            },
            onend: async function (id) {
                await dotnetReference.invokeMethodAsync('OnEndCallback', id);
            },
            onload: async function () {
                await dotnetReference.invokeMethodAsync('OnLoadCallback');
            },
            onloaderror: async function (id, error) {
                await dotnetReference.invokeMethodAsync('OnLoadErrorCallback', id, error);
            },
            onplayerror: async function (id, error) {
                await dotnetReference.invokeMethodAsync('OnPlayErrorCallback', id, error);
            }
        });

        soundId = howl.play();

        howlInstances[soundId] = {
            howl,
            options
        };

        return soundId;
    },
    playSound: function (id) {
        const howl = getHowl(id);
        if (howl) {
            howl.play(soundId);
        }
    },
    stop: function (id) {
        const howl = getHowl(id);
        if (howl) {
            howl.stop();
        }
    },
    pause: function (id) {
        const howl = getHowl(id);
        if (howl) {
            if (howl.playing()) {
                howl.pause(id);
            } else {
                howl.play(soundId);
            }
        }
    },
    seek: function (id, position) {
        const howl = getHowl(id);
        if (howl) {
            howl.seek(position);
        }
    },
    rate: function (id, rate) {
        const howl = getHowl(id);
        if (howl) {
            howl.rate(rate);
        }
    },
    load: function (id) {
        const howl = getHowl(id);
        if (howl) {
            howl.load();
        }
    },
    unload: function (id) {
        const howl = getHowl(id);
        if (howl) {
            howl.unload();

            howlInstances[id] = null;
            delete instances[id];
        }
    },
    getIsPlaying: function (id) {
        const howl = getHowl(id);
        if (howl) {
            return howl.playing();
        }

        return false;
    },
    getRate: function (id) {
        const howl = getHowl(id);
        if (howl) {
            return howl.rate();
        }

        return 0;
    },
    getCurrentTime: function (id) {
        const howl = getHowl(id);
        if (howl && howl.playing()) {
            const seek = howl.seek();
            return seek === Infinity || isNaN(seek) ? null : seek;
        }

        return 0;
    },
    getTotalTime: function (id) {
        const howl = getHowl(id);
        if (howl) {
            const duration = howl.duration();
            return duration === Infinity || isNaN(duration) ? null : duration;
        }

        return 0;
    },
    destroy: function () {
        Object.keys(howlInstances).forEach(key => {
            try {
                unload(key);
            } catch {
                // no-op
            }
        });
    }
};

window.howler = {
    mute: function (muted) {
        Howler.mute(muted);
    },
    getCodecs: function () {
        const codecs = [];
        for (const [key, value] of Object.entries(Howler._codecs)) {
            if (value) {
                codecs.push(key);
            }
        }

        return codecs.sort();
    },
    isCodecSupported: function (extension) {
        return extension ? Howler._codecs[extension.replace(/^x-/, '')] : false;
    }
};

function getHowl(id) {
    return howlInstances[id] ? howlInstances[id].howl : null;
}
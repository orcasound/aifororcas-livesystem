/* Sidebar Toggler */

function ToggleSideBar() {

	$("body").toggleClass("sidebar-toggled");
	$(".sidebar").toggleClass("toggled");
	if ($(".sidebar").hasClass("toggled")) {
		$('.sidebar .collapse').collapse('hide');
	};

	ResizeActivePlayer();
}

/* Modal Map Functionality */

function LoadBingMap(id, latitude, longitude) {
	var map = new Microsoft.Maps.Map(document.getElementById('bingMap-modal-' + id), {
		center: new Microsoft.Maps.Location(latitude, longitude),
		mapTypeId: Microsoft.Maps.MapTypeId.aerial,
		zoom: 12
	});
	CreateScaledPushpin(map.getCenter(), 'img/hydrophone.png', .25, function (pin) {
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

/* Spectrogram Functionality */

var wavesurfer = {}

/* Since resizing skews dimensions of image and player, we'll need to reset each time */

window.onresize = function () {
	ResizeActivePlayer();
}

function ResetPlayButton() {
	$('#play-' + wavesurfer.containerId)
		.removeClass()
		.addClass('fas fa-play-circle fa-3x');
}

function SetPlayButtonToSpinner() {
	$("#play-" + wavesurfer.containerId)
		.removeClass('fa-play-circle')
		.addClass('fa-spinner spinner');
}

function SetSpinnerButtonToPause() {
	$("#play-" + wavesurfer.containerId)
		.removeClass('spinner fa-spinner')
		.addClass('fa-pause-circle');
}

function SetSpinnerButtonToPlay() {
	$("#play-" + wavesurfer.containerId)
		.removeClass('spinner fa-spinner')
		.addClass('fa-play-circle');
}

function SetPlayButtonToPause() {
	$("#play-" + wavesurfer.containerId)
		.removeClass('fa-play-circle')
		.addClass('fa-pause-circle');
}

function SetPauseButtonToPlay() {
	$("#play-" + wavesurfer.containerId)
		.removeClass('fa-pause-circle')
		.addClass('fa-play-circle');
}

function ResetElapsedTime() {
	$("#elapsed-" + wavesurfer.containerId)
		.text('0.00')
}
function SetElapsedTime() {
	$("#elapsed-" + wavesurfer.containerId)
		.text(wavesurfer.getCurrentTime().toFixed(2))
}
function ResetDuration() {
	$("#duration-" + wavesurfer.containerId)
		.text('00.00')
}

function SetDuration() {
	$("#duration-" + wavesurfer.containerId).text(wavesurfer.getDuration().toFixed(2))
}

function SetMaximumVolume() {
	wavesurfer.setVolume(1);
}

function Spectrogram(containerId) {
	var image = $("#spectrogram-" + containerId)[0];

	var width = image.width;
	if (width % 2 != 0) {
		width += 1;
	}

	this.image = image;
	this.height = image.height;
	this.width = width;
}

function AdjustSizes(spectrogram) {
	// This is a funky hack to make the overlay and wavesurfer the correct size
	// When the wave width exceeds the image (should only happen on very large screens)

	if (wavesurfer.drawer.container.clientWidth > spectrogram.width) {
		spectrogram.image.width = wavesurfer.drawer.container.clientWidth;
	}

	wavesurfer.setHeight(spectrogram.image.height);
}

function IsPlayerActive() {
	return (wavesurfer.container != null);
}

function ResizeActivePlayer() {
	if (wavesurfer.container != undefined) {
		spectrogram = new Spectrogram(wavesurfer.containerId);
		AdjustSizes(spectrogram);
	}
}

function DestroyActivePlayer() {

	if (wavesurfer.container != undefined) {
		ResetPlayButton();
		ResetElapsedTime();
		ResetDuration();

		wavesurfer.destroy();

		wavesurfer = {};
	}
}

function InitializeModalSpectrogram(modalId, audioUrl, regionsJson) {

	var containerId = 'modal-' + modalId;

	if (wavesurfer.container != undefined && wavesurfer.containerId != containerId) {
		DestroyActivePlayer();
	}

	var spectrogram = new Spectrogram(containerId);

	wavesurfer = WaveSurfer.create({
		container: '#waveform-' + containerId,
		waveColor: 'rgba(0,0,0,0)',
		progressColor: 'rgba(0,0,0,0)',
		loaderColor: 'purple',
		cursorColor: 'white',
		height: spectrogram.height,
		maxCanvasWidth: spectrogram.width,
		responsive: true,
		fillParent: true,
		plugins: [
			WaveSurfer.regions.create({
				regions: JSON.parse(regionsJson)
			})
		]
	})

	wavesurfer.containerId = containerId;

	SetPlayButtonToSpinner();

	wavesurfer.load(audioUrl);

	wavesurfer.on('ready', function () {
		SetMaximumVolume();
		AdjustSizes(spectrogram);
		SetSpinnerButtonToPlay();
		SetDuration();
	});

	// when something is happening, update
	wavesurfer.on('audioprocess', function () {
		SetElapsedTime();
	});

	// when seeking is used
	wavesurfer.on('seek', function () {
		SetElapsedTime();
		SetPlayButtonToPause();
		wavesurfer.play();
	});
}

function ToggleModalSpectrogram() {

	if (wavesurfer.isPlaying()) {
		SetPauseButtonToPlay();
		wavesurfer.pause();
	}
	else {
		SetPlayButtonToPause();
		wavesurfer.play();
	}
}

function CardSpectrogram(cardId, audioUrl) {

	var containerId = 'card-' + cardId;

	if (wavesurfer.container != undefined && wavesurfer.containerId != containerId) {
		DestroyActivePlayer();
	}

	if (wavesurfer.container == undefined) {

		var spectrogram = new Spectrogram(containerId);

		wavesurfer = WaveSurfer.create({
			container: ('#waveform-' + containerId),
			waveColor: 'rgba(0,0,0,0)',
			progressColor: 'rgba(0,0,0,0)',
			loaderColor: 'purple',
			cursorColor: 'white',
			height: spectrogram.height,
			maxCanvasWidth: spectrogram.width,
			responsive: true,
			fillParent: true
		})

		wavesurfer.containerId = containerId;

		SetPlayButtonToSpinner();

		wavesurfer.load(audioUrl);

		wavesurfer.on('ready', function () {
			SetMaximumVolume();
			AdjustSizes(spectrogram);
			SetSpinnerButtonToPause();
			SetDuration();
			wavesurfer.play();
		});

		// when done playing, reset everything
		wavesurfer.on('finish', function () {
			DestroyActivePlayer();
		});

		// when something is happening, update elapsed time
		wavesurfer.on('audioprocess', function () {
			SetElapsedTime();
		});

		// when seeking is used
		wavesurfer.on('seek', function () {
			SetElapsedTime();
			SetPlayButtonToPause();
			wavesurfer.play();
		});
	}

	else if (wavesurfer.isPlaying()) {
		SetPauseButtonToPlay();
		wavesurfer.pause();
	}

	else {
		SetPlayButtonToPause();
		wavesurfer.play();
	}
}


// Set new default font family and font color to mimic Bootstrap's default styling

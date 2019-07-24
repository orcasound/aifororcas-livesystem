'use strict';

/*
 * Purpose:
 *   Combines all the components of the interface. Creates each component, gets task
 *   data, updates components. When the user submits their work this class gets the workers
 *   annotations and other data and submits to the backend
 * Dependencies:
 *   AnnotationStages (src/annotation_stages.js), PlayBar & WorkflowBtns (src/components.js), 
 *   HiddenImg (src/hidden_image.js), colormap (colormap/colormap.min.js) , Wavesurfer (lib/wavesurfer.min.js)
 * Globals variable from other files:
 *   colormap.min.js:
 *       magma // color scheme array that maps 0 - 255 to rgb values
 *    
 */
function Annotator() {
    this.wavesurfer;
    this.playBar;
    this.annotatortool;
    this.workflowBtns;
    this.currentTask;
    this.taskStartTime;
    // only automatically open instructions modal when first loaded
    this.instructionsViewed = false;
    // Boolean, true if currently sending http post request 
    this.sendingResponse = false;

    this.dataUrl = null;
    // Create color map for spectrogram
    var spectrogramColorMap = colormap({
        colormap: magma,
        nshades: 256,
        format: 'rgb',
        alpha: 1
    });

    // Create wavesurfer (audio visualization component)
    var height = 256;
    this.wavesurfer = Object.create(WaveSurfer);
    this.wavesurfer.init({
        container: '.audio_visual',
        responsive: true,
        waveColor: '#FF00FF',
        progressColor: '#FF00FF',
        cursorColor: '#FFFFFF',
        // For the spectrogram the height is half the number of fftSamples
        fftSamples: height * 2,
        height: height,
        colorMap: spectrogramColorMap
    });

    // Create labels (labels that appear above each region)
    var labels = Object.create(WaveSurfer.Labels);
    labels.init({
        wavesurfer: this.wavesurfer,
        container: '.labels'
    });

    // Create the play button and time that appear below the wavesurfer
    this.playBar = new PlayBar(this.wavesurfer);
    this.playBar.create();

    // Create the annotation stages that appear below the wavesurfer. The stages contain tags 
    // the users use to label a region in the audio clip
    // this.stages = new AnnotationStages(this.wavesurfer, this.hiddenImage);
    // this.stages.create();

    this.annotatortool = new AnnotationReveal(this.wavesurfer);
    this.annotatortool.create();

    // Create Workflow btns (submit and exit)
    this.workflowBtns = new WorkflowBtns();
    this.workflowBtns.create();

    this.addEvents();
}

Annotator.prototype = {
    addWaveSurferEvents: function() {
        var my = this;

        // function that moves the vertical progress bar to the current time in the audio clip
        var updateProgressBar = function () {
            var progress = my.wavesurfer.getCurrentTime() / my.wavesurfer.getDuration();
            my.wavesurfer.seekTo(progress);
        };

        // Update vertical progress bar to the currentTime when the sound clip is 
        // finished or paused since it is only updated on audioprocess
        this.wavesurfer.on('pause', updateProgressBar);
        this.wavesurfer.on('finish', updateProgressBar);    

        // When a new sound file is loaded into the wavesurfer update the  play bar, update the 
        // annotation stages back to stage 1, update when the user started the task, update the workflow buttons.
        // Also if the user is suppose to get hidden image feedback, append that component to the page
        this.wavesurfer.on('ready', function () {
            my.playBar.update();
            my.annotatortool.updateStage(1);
            my.updateTaskTime();
            my.workflowBtns.update();
        });

        this.wavesurfer.on('click', function (e) {
            my.annotatortool.clickDeselectCurrentRegion();
        });
    },

    updateTaskTime: function() {
        this.taskStartTime = new Date().getTime();
    },

    // Event Handler, if the user clicks submit annotations call submitAnnotations
    addWorkflowBtnEvents: function() {
        $(this.workflowBtns).on('submit-annotations', this.submitAnnotations.bind(this));
    },

    addEvents: function() {
        this.addWaveSurferEvents();
        this.addWorkflowBtnEvents();
    },

    // Update the task specific data of the interfaces components
    update: function() {
        var my = this;
        var mainUpdate = function() {

            // Update the different tags the user can use to annotate, also update the solutions to the
            // annotation task if the user is suppose to recieve feedback

            // Update the visualization type and the feedback type and load in the new audio clip
            my.wavesurfer.params.visualization = "spectrogram"; // invisible, spectrogram, waveform            my.wavesurfer.params.feedback = my.currentTask.feedback; // hiddenImage, silent, notify, none 
            my.wavesurfer.load(my.currentTask.uri);

            // my.wavesurfer.on('ready', function () {
            //     $.getJSON(my.currentTask.annotationSolutionsUrl)
            //     .done(function(data) {
            //         my.annotatortool.displayRegions(data);
            //     })
            //     .fail(function() {
            //         alert('Error: Unable to retrieve annotation solution set');
            //     });

            // });

            my.wavesurfer.on('ready', function () {
                    my.annotatortool.displayRegions(my.currentTask.annotations);
            });

        };

        // $.getJSON(this.currentTask.annotationSolutionsUrl)
        // .done(function(data) {
        mainUpdate();
        // })
        // .fail(function() {
        //     alert('Error: Unable to retrieve annotation solution set');
        // });
    },

    // Update the interface with the next task's data
    loadNextTask: function() {
        var my = this;
        $.getJSON(dataUrl)
        .done(function(data) {
            my.currentTask = data;
            my.update();
        })
        .fail(function() {
            alert('Error: Unable to retrieve JSON from Azure blob storage');
        });
    },

    // Collect data about users annotations and submit it to the backend
    submitAnnotations: function() {
        // Check if all the regions have been labeled before submitting
        // if (this.stages.annotationDataValidationCheck()) {
            if (this.sendingResponse) {
                // If it is already sending a post with the data, do nothing
                return;
            }
            this.sendingResponse = true;
            // Get data about the annotations the user has created
            // var content = {
            //     task_start_time: this.taskStartTime,
            //     task_end_time: new Date().getTime(),
            //     visualization: this.wavesurfer.params.visualization,
            //     annotations: this.annotatortool.getAnnotations(),
            //     deleted_annotations: this.annotatortool.getDeletedAnnotations(),
            //     // List of actions the user took to play and pause the audio
            //     play_events: this.playBar.getEvents(),
            // };
            
            var content = {
                uri: this.currentTask.uri,
                absolute_time: this.currentTask.absolute_time,
                source_guid: this.currentTask.source_guid,
                annotations: this.annotatortool.getPodCastAnnotations(),
            };

            this.post(content);
    },

    // Make POST request, passing back the content data. On success load in the next task
    post: function (content) {
        var my = this;
        $.ajax({
            type: 'POST',
            url: $.getJSON(postUrl),
            contentType: 'application/json',
            data: JSON.stringify(content)
        })
        .done(function(data) {
            my.loadNextTask();
        })
        .fail(function() {
            alert('Error: Unable to Submit Annotations');
        })
        .always(function() {
            // No longer sending response
            my.sendingResponse = false;
        });
    }

};

function main() {
    // Create all the components
    var annotator = new Annotator();
    // Load the first audio annotation task
    annotator.loadNextTask();
}
main();

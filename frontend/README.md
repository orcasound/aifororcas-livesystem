# PodCast audio-annotator

### Running the tool

From this directory, i.e. root/frontend, start a python server (python3) with the following command: `python3 -m http.server`. 
Now, browse to `http://localhost:8000/examples/guestannotation.html` in your browser (Edge and Chrome are tested) to view the tool. 

### How to use

The system is currently setup to view and annotate unlabelled master tapes from the WHOIS dataset. See 1:22 at [hackbox page](https://garagehackbox.azurewebsites.net/hackathons/1857/projects/82146) video for how to use. You can continue to reload the page to get new sessions from the classifier. But if you submit, do make sure you do your diligence with the annotations as we haven't setup any user differentiation yet :) 

### Next up 

We plan to hook this up to recent streams from OrcaSound, to both get a sense of the classifier's performance and help improve it in live conditions. If you want to try a different classifier, for now,  let us know. Eventually we're hoping to make it easier to try your own. 

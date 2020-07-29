import * as React from 'react';
import ReactDOM from 'react-dom';

export interface ICandidate {
	annotations: {
		confidence: number,
		duration: number,
		id: number,
		startTime: number,
	}[];
	audioUri: string;
	averageConfidence: number;
	comments: string;
	dateModerated: string;
	detections: number;
	id: string;
	imageUri: string;
	location: {
		name: string,
		longitude: number,
		latitude: number
	};
	moderator: string;
	playerId: string;
	status: string;
	tags: string;
	timestamp: string;
}

export function renderReactDetails(candidate: ICandidate) {
	console.log(candidate);
    const Details = () => {
		if (document.getElementById('react-details-modal') && document.getElementById('react-details-modal')!.style.display === "none"){
			document.getElementById('react-details-modal')!.style.display="block"
		}
        return (
            <div id="react-details-modal" className="overlay">
				<div className="popup">
					<h2>{candidate.id}</h2>
					<span className="close" onClick={() => document.getElementById('react-details-modal')!.style.display="none"}>&times;</span>
					<div className="content container">
						<div className="row">
							<div className="col-3">
								<p className="orca-data-strip-data">
									<i className="oi oi-clock" style={{width: "20px"}}></i>&nbsp;
									{candidate.timestamp}
								</p>
							</div>
							<div className="col-3">
								<p className="orca-data-strip-data">
									<i className="oi oi-map-marker" style={{width: "20px"}}></i>&nbsp;
									{candidate.location.name}
								</p>
							</div>
							<div className="col-3">
								<p className="orca-data-strip-data">
									<i className="oi oi-microphone" style={{width: "20px"}}></i>&nbsp;
									{candidate.detections} detection(s)
								</p>
							</div>
							<div className="col-3">
								<p className="orca-data-strip-data">
									<i className="oi oi-graph" style={{width: "20px"}}></i>&nbsp;
									{candidate.averageConfidence}% average confidence
								</p>
							</div>
						</div>
						<div className="row">
							<img src="https://via.placeholder.com/700x400"></img>
						</div>
					</div>
				</div>
			</div>
        )
    }

    ReactDOM.render(Details(), document.getElementById(`react-details`));
}
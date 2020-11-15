import * as React from 'react';
import ReactDOM from 'react-dom';
import { ReactAnnotationViewer } from './ReactAnnotationViewer';

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
	whaleTime: string;
}

export function renderReactDetails(candidate: ICandidate) {
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
									{candidate.whaleTime}
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
						<div className="row" style={{marginTop: "30px"}}>
							<ReactAnnotationViewer
								imageUri={candidate.imageUri}
								width={640}
								height={240}
								predictions={candidate.annotations}
								audioUri={candidate.audioUri}
							/>
						</div>
						<div className="row" style={{marginTop: "30px"}}>
							<div className="col-6">
								<label className="mb-0">Was there an SRKW call in this clip?</label>
								<div style={{display: "grid"}}>
									<label>
										<input type="radio"
										   name="found"
										   onChange={() => { console.log("implement") }}
										   style={{marginRight: "10px"}}
										/>
										{"Yes"}
									</label>
									<label>
										<input type="radio"
											   name="found"
											   onChange={() => { console.log("implement") }}
											   style={{marginRight: "10px"}}
											/>
										{"No"}
									</label>
									<label>
										<input type="radio"
										   name="found"
										   onChange={() => { console.log("implement") }}
										   style={{marginRight: "10px"}}
										/>
										{"Don't know"}
									</label>
    	                		</div>
							</div>
							<div className="col-6">
								<div className="form-group">
									<label>Tags</label>
									<div>
										<input className="form-control" placeholder="Add tags" />
									</div>
								</div>
								<div className="form-group">
									<label>Comments</label>
									<div>
										<input className="form-control" placeholder="Add comments" />
									</div>
								</div>
							</div>
						</div>
						<div className="row">
							<button className="btn btn-primary">Submit</button>
						</div>
					</div>
				</div>
			</div>
        )
    }

    ReactDOM.render(Details(), document.getElementById(`react-details`));
}
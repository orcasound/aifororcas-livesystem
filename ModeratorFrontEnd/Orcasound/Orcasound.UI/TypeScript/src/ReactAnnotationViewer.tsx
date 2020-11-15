import * as React from "react";
const Annotorious = require("@recogito/annotorious");

interface IPrediction {
    id: number,
    startTime: number,
    duration: number,
    confidence: number
}

interface IReactAnnotationViewerProps {
    imageUri: string,
    audioUri: string
    width: number,
    height: number
    predictions: IPrediction[]
}

export class ReactAnnotationViewer extends React.Component<IReactAnnotationViewerProps> {
    private _anno: any = null;

    public componentDidMount() {
        this._anno = Annotorious.init({
            image: "image-to-annotate",
            readOnly: true,
        });

        for (let i = 0; i < this.props.predictions.length; i++) {
            const pred = this.props.predictions[i];

            let x = (pred.startTime * this.props.width) / 60;
            if (pred.startTime > 55) {
                x = 55 * this.props.width / 60;
            }

            const y = 10;
            const w = (pred.duration * this.props.width) / 60;
            const h = this.props.height - y - 10;

            const annotation = {
                "@context": "http://www.w3.org/ns/anno.jsonld",
                id: i,
                type: "Annotation",
                body: [
                    {
                        type: "TextualBody",
                        value: `${pred.confidence*100}%`,
                    },
                ],
                target: {
                    selector: {
                        type: "FragmentSelector",
                        conformsTo: "http://www.w3.org/TR/media-frags/",
                        value: `xywh=pixel:${x},${y},${w},${h}`,
                    },
                },
            };

            this._anno.addAnnotation(annotation);
        }
    }

    public render(): JSX.Element {
        return (
            <div style={{display: "inline-grid"}}>
                <img
                    id="image-to-annotate"
                    src={this.props.imageUri}
                    style={{
                        width: this.props.width,
                        height: this.props.height,
                    }}
                />
                <audio src={this.props.audioUri} controls />
            </div>
        );
    }
}

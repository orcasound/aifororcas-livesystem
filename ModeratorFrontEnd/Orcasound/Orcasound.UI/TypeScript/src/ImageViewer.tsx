import * as React from "react";
const Annotorious = require("@recogito/annotorious");

interface Prediction {
    id: number,
    startTime: number,
    duration: number,
    confidence: number
}

interface IImageViewer {
    imageUri: string,
    width: number,
    height: number
    predictions: Prediction[]
}

export class ImageViewer extends React.Component<IImageViewer> {
    anno: any = null;
    predictions = [
        {id: 0, startTime: 10.8, duration: 2.45, confidence: 0.541}
    ];
    loadAnnotations() {
        for (let i = 0; i < this.props.predictions.length; i++) {
            const pred = this.props.predictions[i]

            console.log(pred.startTime)
            console.log(pred.duration)
            console.log(pred.confidence)

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
            this.anno.addAnnotation(annotation);

        }
    }
    componentDidMount() {
        this.anno = Annotorious.init({
            image: "image-to-annotate",
            readOnly: true,
        });
        this.loadAnnotations();
    }
    render() {
        return (
            
                <img
                    id="image-to-annotate"
                    src={this.props.imageUri}
                    style={{
                        width: this.props.width,
                        height: this.props.height
                    }}
                />
            
        );
    }
}

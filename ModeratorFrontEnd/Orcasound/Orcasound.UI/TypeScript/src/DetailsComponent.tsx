import * as React from "react";
import ReactDOM from "react-dom";
import { ImageViewer } from "./ImageViewer"

export function renderDetailsComponent(ImageUri: string, Annotations: any) {
    console.log("Hello");

    const DetailsComponent = () => {
        return (
            <div>
                <div
                    style={{
                        backgroundColor: "rgba(20, 20, 20, 0.8)",
                        position: "absolute",
                        top: "0",
                        left: "0",
                        width: "100%",
                        height: "100%",
                    }}
                ></div>
                <div
                style={{
                    position: "absolute",
                    top: "0",
                    left: "0",
                    display: "flex",
                    width: "100%",
                    height: "100%",
                    margin: "auto",
                    justifyContent: "center",
                    alignItems: "center"
                }}>

                    <ImageViewer imageUri={ImageUri} width={640} height={240} predictions={Annotations}/>
                </div>  
            </div>
        );
    };

    ReactDOM.render(
        DetailsComponent(),
        document.getElementById("details-view")
    );
}

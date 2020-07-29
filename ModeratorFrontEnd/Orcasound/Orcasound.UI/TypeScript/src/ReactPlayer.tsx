import * as React from 'react';
import ReactDOM from 'react-dom';

export function renderReactPlayer(imgSrc: string, audioSrc: string, playerId: string) {
    const Player = () => {
        return (
            <video
                controls
                width="280px"
                height="200px"
                poster={imgSrc}
                style={{
                  borderRadius: "5px",
                  backgroundColor: "black"
                }}
                src={audioSrc}
            >
                <p>
                    To view this clip please enable JavaScript, and consider upgrading to a
                    newer version of the web browser.
                </p>
            </video>
        )
    }

    ReactDOM.render(Player(), document.getElementById(`react-player-${playerId}`));
}
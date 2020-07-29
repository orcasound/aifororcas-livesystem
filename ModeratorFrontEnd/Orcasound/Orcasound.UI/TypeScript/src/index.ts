import { renderReactPlayer } from './ReactPlayer';
import { renderReactDetails, ICandidate } from './ReactDetails';

export function RenderReactPlayer(imgSrc: string, audioSrc: string, playerId: string) {
    return renderReactPlayer(imgSrc, audioSrc, playerId);
}

export function RenderReactDetails(candidate: ICandidate) {
    return renderReactDetails(candidate);
}
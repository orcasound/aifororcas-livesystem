import { renderTestComponent } from './AudioPlayer';
import { renderDetailsComponent } from './DetailsComponent';

export function RenderTestComponent() {
    return renderTestComponent();
}

export function RenderDetailsComponent(ImageUri: string, Annotations: any) {
    return renderDetailsComponent(ImageUri, Annotations);
}
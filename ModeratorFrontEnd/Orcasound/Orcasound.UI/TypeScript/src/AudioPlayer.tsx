import * as React from 'react';
import ReactDOM from 'react-dom';

export function renderTestComponent() {
    const TestComponent = () => <div>Hello React</div>

    ReactDOM.render(TestComponent(), document.getElementById('react-test'));
}
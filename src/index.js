import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { autoStartAgent } from './utils/agentLauncher';

// Start the local agent as the app launches, so it is already up by the time
// the user reaches the Flasher. Silent and non-blocking - it renders nothing
// and the app does not wait on it.
autoStartAgent();

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);



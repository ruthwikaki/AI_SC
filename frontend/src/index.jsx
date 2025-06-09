import React from 'react';
import './config/agGridSetup';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css'; // Make sure this file exists, if not, create it

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

import React, { useState, useEffect } from 'react';
import '../styles/bluetoothConnection.css';

const BluetoothIcon = () => (
  <img src="../assets/bluetooth-off.png" alt="Bluetooth" width="28" height="28" />
);

const BluetoothSearchingIcon = () => (
  <img src="../assets/bluetooth-searching.png" alt="Bluetooth Searching" width="28" height="28" />
);

const BluetoothConnectedIcon = () => (
  <img src="../assets/bluetooth-connected.png" alt="Bluetooth Connected" width="28" height="28" />
);

const CloseIcon = () => (
  <img src="../assets/close.png" alt="Close" width="20" height="20" />
);

const BluetoothConnection = ({ connectedDevice, setConnectedDevice}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [state, setState] = useState('off');
  const [devices, setDevices] = useState([]);
  const [connectingDeviceId, setConnectingDeviceId] = useState(null);
  const [localConnectedDevice, setLocalConnectedDevice] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const activeDevice = connectedDevice || localConnectedDevice;

  


  const fetchBluetoothDevices = async () => {
    setIsSearching(true);
    setErrorMessage('');

    try {
      const storedDevice = localStorage.getItem('connectedDevice');

      if (storedDevice) {
        const parsedDevice = JSON.parse(storedDevice);
        setConnectedDevice(parsedDevice);
        setState('connected');
        setIsSearching(false);
        return;
      }

      const response = await fetch('/api/bluetooth/devices');

      if (!response.ok) {
        if (response.status === 500) {
          throw new Error('Your system Bluetooth is off. Please turn on Bluetooth and try again.');
        } else {
          throw new Error(`Server responded with status: ${response.status}`);
        }
      }

      const data = await response.json();

      if (Array.isArray(data)) {
        setDevices(data);
      } else {
        throw new Error('Invalid data format received from server');
      }
    } catch (error) {
      console.error('Error fetching devices:', error);
      setErrorMessage(error.message);
      setDevices([]);
    } finally {
      setIsSearching(false);
    }
  };
  const connectToDevice = async (device) => {
    if (connectingDeviceId === device.id) {
      return;
    }
    setConnectingDeviceId(device.id);
    setErrorMessage('');

    try {
      const response = await fetch('/api/bluetooth/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deviceId: device.id }),
      });

      const result = await response.json();

      if (result.success) {
        setConnectedDevice(device);
        setState('connected');
        localStorage.setItem('connectedDevice', JSON.stringify(device));
        setIsOpen(false);
      } else {
        throw new Error(result.message || 'Connection failed');
      }
    } catch (error) {
      console.error('Connection error:', error);
      setErrorMessage(error.message);
      setConnectingDeviceId(null);
      setState('searching');
      setDevices([]);
    } finally {
      if (connectingDeviceId === device.id) {
        setConnectingDeviceId(null);
      }
    }
  };

  const disconnectDevice = async () => {
    // Pull the latest device info directly from localStorage
    const storedDevice = localStorage.getItem('connectedDevice');
    const device = storedDevice ? JSON.parse(storedDevice) : null;
  
    if (!device) {
      return;
    }
  
    try {
      setLocalConnectedDevice(null);
      setConnectedDevice(null);
      setState('off');
      localStorage.removeItem('connectedDevice');
      setIsOpen(false);
  
      // Send disconnect request to backend
      const response = await fetch('/api/bluetooth/disconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deviceId: device.id }),
      });
  
      const result = await response.json();
  
      if (response.ok) {
        console.log("Device disconnected:", result.message);
      } else {
        setErrorMessage(result.message || 'Failed to disconnect');
      }
    } catch (error) {
      setErrorMessage(`Disconnect failed: ${error.message}`);
    }
  };
  


  useEffect(() => {
    const restorePreviousConnection = async () => {
      const storedDevice = localStorage.getItem('connectedDevice');

      if (storedDevice) {
        const parsedDevice = JSON.parse(storedDevice);

        try {
          const response = await fetch('/api/bluetooth/status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ deviceId: parsedDevice.id }),
          });

          const result = await response.json();

          if (result.connected) {
            setConnectedDevice(parsedDevice);
            setLocalConnectedDevice(parsedDevice);
            setState('connected');
          } else {
            localStorage.removeItem('connectedDevice');
          }
        } catch (error) {
          console.error('Error checking connection status:', error);
          if (error.message && error.message.includes('500')) {
            setErrorMessage('Your system Bluetooth is off. Please turn on Bluetooth.');
          }
        }
      }
    };

    restorePreviousConnection();
  }, []);


  const toggleBluetooth = () => {
    if (state === 'off') {
      setIsOpen(true);
      setState('searching');
      setDevices([]);
      setErrorMessage('');
      setConnectedDevice(null);
      setConnectingDeviceId(null);
      fetchBluetoothDevices();
    } else if (state === 'connected') {
      disconnectDevice();
    } else {
      setIsOpen(false);
      setState('off');
      setDevices([]);
      setConnectedDevice(null);
      setConnectingDeviceId(null);
    }
  };


  const renderIcon = () => {
    if (state === 'off') return <BluetoothIcon />;
    if (state === 'searching') return <BluetoothSearchingIcon />;
    if (state === 'connected') return <BluetoothConnectedIcon />;
  };

  const renderButtonText = () => {
    if (state === 'off') return "Turn On Bluetooth";
    if (state === 'searching') return "Searching...";
    if (state === 'connected') return `Connected to ${activeDevice?.name || 'device'}`;
  };
  
  
  

  return (
    <div className="bluetooth-container">
      <button
        onClick={toggleBluetooth}
        className={`bluetooth-button ${state}`}
        title={state === 'connected' ? `Connected to ${connectedDevice?.name}` : ''}
      >
        {renderIcon()}
        <span>{renderButtonText()}</span>
      </button>

      <div className={`bluetooth-dropdown ${isOpen ? 'open' : ''}`}>
        <div className="dropdown-header">
          <span>Available Devices</span>
          <button
            onClick={() => setIsOpen(false)}
            className="close-button"
          >
            <CloseIcon />
          </button>
        </div>

        <div className="devices-container">
          {errorMessage && <div className="error-message">{errorMessage}</div>}

          {devices.length === 0 && !isSearching && !errorMessage && (
            <div className="no-devices">
              <p>No devices found</p>
              <button
                className="scan-button"
                onClick={fetchBluetoothDevices}
                disabled={isSearching}
              >
                Scan Again
              </button>
            </div>
          )}

          {isSearching && (
            <div className="searching-indicator">
              <span>Searching for devices...</span>
            </div>
          )}

          {devices.map(device => (
            <div
              key={device.id}
              className={`device-item ${activeDevice?.id === device.id ? 'connected' : ''}`}
            >
              <div className="device-info">
                <span>{device.name}</span>
              </div>

              {connectingDeviceId === device.id ? (
                <div className="connecting-indicator">
                  <span>Connecting...</span>
                </div>
              ) : connectedDevice?.id === device.id ? (
                <BluetoothConnectedIcon />
              ) : (
                <button onClick={() => connectToDevice(device)} className="connect-button">
                  Connect
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default BluetoothConnection;

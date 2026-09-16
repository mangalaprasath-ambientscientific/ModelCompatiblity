import React, { useEffect, useState, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import io from 'socket.io-client';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

ChartJS.defaults.font.family = '"Gilroy-Medium", sans-serif';

const RealTimeGraph = ({ isGraphActive, prepareSocket, duration, currentSample, onSocketStatusChange , disconnectSensor  }) => {
  // Socket and connection state
  const [socket, setSocket] = useState(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const connectionAttemptRef = useRef(false);
  const socketRef = useRef(null);
  
  // Sensor data state (accelerometer and gyroscope)
  const [accelX, setAccelX] = useState([{x: 0, y: 0}]);
  const [accelY, setAccelY] = useState([{x: 0, y: 0}]);
  const [accelZ, setAccelZ] = useState([{x: 0, y: 0}]);
  const [gyroX, setGyroX] = useState([{x: 0, y: 0}]);
  const [gyroY, setGyroY] = useState([{x: 0, y: 0}]);
  const [gyroZ, setGyroZ] = useState([{x: 0, y: 0}]);
  
  // Graph visualization state
  const pointCounter = useRef(0);
  const visibleWindowStart = useRef(0);
  
  // Calculate maximum points based on duration and sampling rate (20Hz)
  const MAX_POINTS = duration * 20;


  const getOptions = () => {
    const currentPoint = pointCounter.current;
    const minX = currentPoint > MAX_POINTS ? currentPoint - MAX_POINTS : 0;
    const maxX = Math.max(MAX_POINTS, currentPoint);
    
    return {
      responsive: true,
      animation: {
        duration: 0  // Disable animations for performance
      },
      interaction: {
        intersect: false,
        mode: 'index'
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#ECECEC',
            font: {
              size: 14
            },
            padding: 30
          }
        },
        tooltip: {
          callbacks: {
            title: (context) => `Data Point ${context[0].parsed.x}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Data Points',
            color: '#ECECEC',
            font: {
              size: 14,
              weight: 'bold'
            },
            padding: {
              top: 0,
              bottom: 0
            }
          },
          min: minX,
          max: maxX,
          grid: {
            color: 'rgba(144, 164, 174, 0.3)'
          },
          ticks: {
            color: '#ECECEC',
            stepSize: Math.ceil(MAX_POINTS / 10),
          }
        },
        y: {
          display: true,
          min: -35000,
          max: 35000,
          grid: {
            color: 'rgba(144, 164, 174, 0.3)'
          },
          ticks: {
            stepSize: 16500,
            color: '#ECECEC',
            callback: (value) => Math.round(value)
          },
          title: {
            display: true,
            text: 'Amplitude',
            color: '#ECECEC',
            font: {
              size: 14,
              weight: 'bold'
            },
            padding: {
              top: 10,
              bottom: 0
            }
          }
        }
      }
    };
  };

  const connectSocketIO = () => {
    if (connectionAttemptRef.current) return null;
    connectionAttemptRef.current = true;
    console.log("Attempting socket connection...");
      
    const newSocket = io("http://127.0.0.1:9262", {
      transports: ["polling"],
      withCredentials: true,
      upgrade: false,
      reconnectionAttempts: 3,    // Try to reconnect 3 times
      reconnectionDelay: 500,     // Wait 500ms between attempts
      timeout: 5000               // Give up after 5 seconds
    });

    // Handle successful connection
    newSocket.on("connect", () => {
      console.log("SENSOR SOCKET ON");
      newSocket.emit("connect_sensor");
      connectionAttemptRef.current = false;
      socketRef.current = newSocket;
      setSocketConnected(true);
      resetGraphData();
    });

    // Handle connection errors
    newSocket.on("connect_error", (error) => {
      console.error("Socket connection error:", error);
      connectionAttemptRef.current = false;
    });

    // Process incoming IMU data
    newSocket.on("receive_imu_gyro", (data) => {
      try {
        pointCounter.current += 1;
        const currentPoint = pointCounter.current;
        
        if (data.accelerometer) {
          updateDataArray(setAccelX, data.accelerometer.x, currentPoint);
          updateDataArray(setAccelY, data.accelerometer.y, currentPoint);
          updateDataArray(setAccelZ, data.accelerometer.z, currentPoint);
        }

        if (data.gyroscope) {
          updateDataArray(setGyroX, data.gyroscope.x, currentPoint);
          updateDataArray(setGyroY, data.gyroscope.y, currentPoint);
          updateDataArray(setGyroZ, data.gyroscope.z, currentPoint);
        }

        // Update visible window for sliding graph effect
        if (currentPoint > MAX_POINTS) {
          visibleWindowStart.current = currentPoint - MAX_POINTS;
        }
      } catch (error) {
        console.error("Error processing data:", error);
      }
    });

    // Handle disconnection
    newSocket.on("disconnect_sensor", () => {
      console.log("Socket disconnected");
      connectionAttemptRef.current = false;
      socketRef.current = null;
      setSocketConnected(false);
      resetGraphData();
    });
      
    setSocket(newSocket);
    return newSocket;
  };

  /**
   * Notify parent component about socket connection status changes
   */
  useEffect(() => {
    if (onSocketStatusChange && socketConnected !== undefined) {
      onSocketStatusChange(socketConnected);
    }
  }, [socketConnected, onSocketStatusChange]);

  /**
   * Updates a data array with a new sensor value
   * Maintains a sliding window of data points for efficient rendering
   * 
   * @param {Function} setterFunction - React state setter function
   * @param {number} newValue - New sensor value to add
   * @param {number} pointIndex - Current data point index
   */
  const updateDataArray = (setterFunction, newValue, pointIndex) => {
    setterFunction(prev => {
      const newPoint = {x: pointIndex, y: newValue};
      const newArray = [...prev, newPoint];

      // Trim array if it grows too large
      if (newArray.length > MAX_POINTS + 10) {
        return newArray.filter(point => point.x >= visibleWindowStart.current);
      }
      
      return newArray;
    });
  };

  /**
   * Resets all graph data to initial state
   * Called when connection is established or recording is stopped
   */
  const resetGraphData = () => {
    const initialPoint = {x: 0, y: 0};
    setAccelX([initialPoint]);
    setAccelY([initialPoint]);
    setAccelZ([initialPoint]);
    setGyroX([initialPoint]);
    setGyroY([initialPoint]);
    setGyroZ([initialPoint]);
    pointCounter.current = 0;
    visibleWindowStart.current = 0;
  };

  /**
   * Effect to handle socket preparation before recording starts
   * Triggered when countdown reaches 1
   */
  useEffect(() => {
    if (prepareSocket && !socket && !connectionAttemptRef.current) {
      console.log("Preparing socket connection as countdown reaches 1...");
      const newSocket = connectSocketIO();
      setSocket(newSocket);
    }
  }, [prepareSocket]);

  /**
   * Effect to handle socket lifecycle based on graph active state
   * Connects when graph becomes active, disconnects when inactive
   */
  useEffect(() => {
    if (isGraphActive && !socket && !connectionAttemptRef.current) {
      console.log("Socket Instance initiated");
      const newSocket = connectSocketIO();
      setSocket(newSocket);
    } 

    // Cleanup function for component unmount
    return () => {
      if (socket) {
        console.log("Cleanup: disconnecting socket");
        socket.emit("disconnect_sensor");
        socket.disconnect();
        connectionAttemptRef.current = false;
        socketRef.current = null;
      }
    };
  }, [isGraphActive]);

  /**
   * Reset graph data when duration changes
   */
  useEffect(() => {
    if (duration) {
      resetGraphData();
    }
  }, [duration]);


  useEffect(() => {
    if (socket && disconnectSensor !== undefined) {
      console.log("Disconnecting sensor from prop change");
      socket.emit("disconnect_sensor");
      socket.disconnect();
      setSocket(null);
      setSocketConnected(false);
      connectionAttemptRef.current = false;
      socketRef.current = null;
      resetGraphData();
    }
  }, [disconnectSensor]);
  
  /**
   * Chart data configuration for Chart.js
   * Defines datasets for accelerometer and gyroscope data with styling
   */
  const chartData = {
    datasets: [
      {
        label: 'Accel-X',
        data: accelX,
        borderColor: 'rgb(255, 0, 0)',
        backgroundColor: 'rgba(255, 0, 0, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Accel-Y',
        data: accelY,
        borderColor: 'rgb(0, 255, 0)',
        backgroundColor: 'rgba(0, 255, 0, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Accel-Z',
        data: accelZ,
        borderColor: 'rgb(0, 0, 255)',
        backgroundColor: 'rgba(0, 0, 255, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Gyro-X',
        data: gyroX,
        borderColor: 'rgb(255, 165, 0)',   // Orange
        backgroundColor: 'rgba(255, 165, 0, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Gyro-Y',
        data: gyroY,
        borderColor: 'rgb(0, 128, 0)',   // Green
        backgroundColor: 'rgba(0, 128, 0, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Gyro-Z',
        data: gyroZ,
        borderColor: 'rgb(0, 255, 255)',   // Cyan
        backgroundColor: 'rgba(0, 255, 255, 0.5)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1
      }
    ]
  };

  return (
    <div className='real-time-graph' style={{ backgroundColor: '#202020' }}>
      <div className="Sensor-tiltle"><h1>MPU-6050 Real-time Data</h1></div>
      <Line options={getOptions()} data={chartData} style={{height: "75%" }} />
    </div>
  );
};

export default RealTimeGraph;
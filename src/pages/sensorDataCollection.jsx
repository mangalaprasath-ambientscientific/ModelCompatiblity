import React, { useState, useEffect } from "react";
import Button from "../components/Button";
import RealTimeGraph from "../components/RealTimeGraph";
import PageTransition from "../components/PageTransition";
import ThreeJSVisualizer from "../components/ThreeJSVizulaizer";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper";
import "../styles/sensordatacollection.css";
import BluetoothConnection from "../components/BluetoothConnection";


const SensorDataCollection = () => {
  const [isGraphActive, setIsGraphActive] = useState(false);
  const [buttonText, setButtonText] = useState("Start");
  const [viewMode, setViewMode] = useState('Graph');
  const [connectedDevice, setConnectedDevice] = useState(null);

  const [countdownState, setCountdownState] = useState("idle");
  const [countdown, setCountdown] = useState(0);
  const [currentSample, setCurrentSample] = useState(0);
  const [totalSamples, setTotalSamples] = useState(0);
  const [sampleDuration, setSampleDuration] = useState(0);
  const [prepareSocket, setPrepareSocket] = useState(false);
  const [socketConnected, setSocketConnected] = useState(false);
  const [disconnectSensor, setDisconnectSensor] = useState(false);

  const [inputValues, setInputValues] = useState({
    person_name: "",
    word_name: "",
    number_of_samples: "",
    select_device: "IMU - MPU6050",
    select_g: "2g / 250° per sec",
    duration: "",
  });



  const handleSocketStatusChange = (status, error) => {
    if (error?.includes("The object has been closed")) {
      toast.warning("Something went wrong, reset device");

      // Reset all states to initial values
      setIsGraphActive(false);
      setPrepareSocket(false);
      setCountdownState("idle");
      setButtonText("Start");
      setSocketConnected(false);
      setDisconnectSensor(false);
      setCountdown(0);
      setCurrentSample(0);
      setTotalSamples(0);
      setSampleDuration(0);
      setInputValues({
        person_name: "",
        word_name: "",
        number_of_samples: "",
        select_device: "IMU - MPU6050",
        select_g: "2g / 250° per sec",
        duration: "",
      });
    } else {
      setSocketConnected(status);
    }
  };

  useEffect(() => {
    let timer;

    if (countdownState === "idle") {
      return;
    }

    if (countdownState === "getReady") {
      setButtonText("Running...");

      if (countdown === 0) {
        setCountdown(1); // Start with Step 1
      } else if (countdown < 3) {
        // Move to next step every 2 seconds
        timer = setTimeout(() => {
          setCountdown(countdown + 1);
        }, 2000);
      } else {
        // After 6 seconds, move to countdown
        timer = setTimeout(() => {
          setCountdownState("counting");
          setCountdown(3);
        }, 2000);
      }
    }

    else if (countdownState === "counting") {
      if (countdown > 1) {
        timer = setTimeout(() => {
          setCountdown(countdown - 1);
        }, 1000);
      } else if (countdown === 1) {
        timer = setTimeout(() => {
          setPrepareSocket(true);
          setCountdownState("recording");
          setIsGraphActive(true);

          setTimeout(() => {
            if (currentSample < totalSamples - 1) {
              setCurrentSample((prevSample) => prevSample + 1);
              setCountdown(0);
              setCountdownState("getReady");
            } else {
              setCountdownState("excel");
              setDisconnectSensor((prev) => !prev);
              setTimeout(() => {
                toast.success("Data collection Done! Check result Logs");
                setIsGraphActive(false);
                setPrepareSocket(false);
                setCountdownState("idle");
                setButtonText("Start");
                setInputValues({
                  person_name: "",
                  word_name: "",
                  number_of_samples: "",
                  select_device: "IMU - MPU6050",
                  select_g: "2g / 250° per sec",
                  duration: "",
                });
              }, 9000);
            }
          }, sampleDuration * 1000);
        }, 1000);
      }
    }

    return () => clearTimeout(timer);
  }, [countdownState, countdown, currentSample, totalSamples, sampleDuration, disconnectSensor]);


  const handleInputChange = (event) => {
    const { id, value } = event.target;
    setInputValues((prevValues) => ({
      ...prevValues,
      [id]: value,
    }));
  };

  const handleKeyDown = (e, nextInputId) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // Prevent form submission
      const nextInput = document.getElementById(nextInputId);
      if (nextInput) {
        nextInput.focus();
      }
    }
  };


  const handleInputBlur = (event) => {
    const { id, value } = event.target;
    const parsedValue = parseInt(value);

    if (id === "duration" && parsedValue < 3) {
      toast.error("Duration must be at least 3 seconds.");
      setInputValues((prevValues) => ({ ...prevValues, [id]: "" }));
      return;
    }

    if (id === "number_of_samples" && parsedValue <= 0) {
      toast.error("Please enter a positive number greater than 0.");
      setInputValues((prevValues) => ({ ...prevValues, [id]: "" }));
      return;
    }
  };



  const handleStartStop = () => {
    const { person_name, word_name, number_of_samples, select_device, select_g, duration } = inputValues;

    if (!connectedDevice) {
      toast.error("Please connect to Bluetooth first!");
      return;
    }

    if (countdownState !== "idle") {
      setDisconnectSensor(!disconnectSensor);
      setIsGraphActive(false);
      setPrepareSocket(false);
      setCountdownState("idle");
      setButtonText("Start");

      setInputValues({
        person_name: "",
        word_name: "",
        number_of_samples: "",
        select_device: "IMU - MPU6050",
        select_g: "2g / 250° per sec",
        duration: "",
      });
      return;
    }

    if (!person_name || !word_name || !number_of_samples || !select_device || !select_g || !duration) {
      toast.error("Please fill in all input fields.");
      return;
    }

    if (isNaN(parseInt(number_of_samples)) || parseInt(number_of_samples) <= 0) {
      toast.error("Please enter a valid number of samples.");
      return;
    }

    if (isNaN(parseInt(duration)) || parseInt(duration) < 3) {
      toast.error("Duration must be at least 3 seconds.");
      return;
    }

    setTotalSamples(parseInt(number_of_samples));
    setSampleDuration(parseInt(duration));
    setCurrentSample(0);
    setCountdownState("getReady");
    setButtonText("Running...");

    handleSubmit();
  };


  const handleSubmit = async () => {
    try {
      console.log("Data submitted successfully!")
      await fetch("http://127.0.0.1:9262/sensordata", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputValues),
      });
    } catch (error) {
      toast.error("Error occurred while submitting data.");
    }
  };


  const renderCountdownMessage = () => {
    if (countdownState === "idle") {
      return null;
    }

    if (countdownState === "getReady") {
      let getReadyMessages = [
        "Get your device ready!",
        "Position the sensor correctly!",
        "Stay steady, starting soon in...!",
      ];

      return (
        <div className="countdown-message">
          <h2>{getReadyMessages[countdown - 1]}</h2>
          <p>Sample {currentSample + 1} of {totalSamples}</p>
        </div>
      );
    }


    if (countdownState === "counting") {
      return (
        <div className="countdown-message">
          <h1>{countdown}</h1>
        </div>
      );
    }

    if (countdownState === "excel") {
      return (
        <div className="countdown-message">
          <h2>Consolidating data...</h2>
        </div>
      );
    }

  };

  return (
    <div>
      <ToastWrapper />
      <div className="container-sensor">
        <div className="left-part-animation-sensor">
        <div className="circle-buttons">
            <div className="nav-tooltip-wrapper">
              <PageTransition to="/audiodatacollection">
                <img
                  src="../assets/audio.png"
                  alt="Audio Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
              <div className="nav-tooltip info">Audio</div>
            </div>

            <div className="nav-tooltip-wrapper">
              <PageTransition to="/videodatacollection">
                <img
                  src="../assets/vision.png"
                  alt="Video Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
              <div className="nav-tooltip info">Vision</div>
            </div>

            <div className="nav-tooltip-wrapper">
              <PageTransition to="/sensordatacollection">
                <img
                  src="../assets/sensor_fusion.png"
                  alt="Sensor Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
              <div className="nav-tooltip info">Sensor</div>
            </div>
          </div>

          <div className="live-graph">
            {renderCountdownMessage()}

            {/* <div className="graph-toggle-button" onClick={() => setViewMode(viewMode === '3D' ? 'Graph' : '3D')}>
              {viewMode === '3D' ? (
                <img src="../assets/line-graph-2.png" alt="Graph Icon" className="graph-icon" />
              ) : (
                "3D"
              )}
            </div> */}
            <div style={{ display: viewMode === '3D' ? 'block' : 'none' }}>
              <ThreeJSVisualizer isGraphActive={isGraphActive} viewMode={viewMode} />
            </div>
            <div style={{ display: viewMode === '3D' ? 'none' : 'block' }}>
              <RealTimeGraph
                isGraphActive={isGraphActive}
                prepareSocket={prepareSocket}
                duration={inputValues.duration}
                currentSample={currentSample}
                onSocketStatusChange={handleSocketStatusChange}
                disconnectSensor={disconnectSensor}
              />
            </div>
          </div>
        </div>

        <div className="right-part-sc">
          <BluetoothConnection
            connectedDevice={connectedDevice}
            setConnectedDevice={setConnectedDevice}
          />
          <div className="upload-section">
            <p className="upload-text">Collect Samples from Sensors</p>
            <div className="upload-line-sc"></div>
          </div>
          <form className="input-form-sensor" onSubmit={(e) => e.preventDefault()}>
            <div className="sensor-tooltip-wrapper">
              <input
                type="text"
                spellCheck="false"
                id="person_name"
                className="input-box-sensor"
                placeholder="Person Name"

                value={inputValues.person_name}
                onKeyDown={(e) => handleKeyDown(e, "word_name")}
                onChange={handleInputChange}
                disabled={countdownState !== "idle"}
              />
              <div className="sensor-tooltip info">Enter the name of the person whose data is being recorded</div>
            </div>

            <div className="sensor-tooltip-wrapper">
              <input
                type="text"
                id="word_name"
                spellCheck="false"
                className="input-box-sensor"
                placeholder="Class Name"
                value={inputValues.word_name}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "number_of_samples")}
                disabled={countdownState !== "idle"}
              />
              <div className="sensor-tooltip info">Enter the class or label name for this sample</div>
            </div>

            <div className="sample-rate-duration">
              <div className="sensor-tooltip-wrapper">
                <input
                  type="number"
                  id="number_of_samples"
                  className="input-box-sensor-one"
                  placeholder="Number of Samples"
                  min="1"
                  value={inputValues.number_of_samples}
                  onWheel={(event) => event.currentTarget.blur()}
                  onChange={handleInputChange}
                  onBlur={handleInputBlur}
                  onKeyDown={(e) => handleKeyDown(e, "duration")}
                  disabled={countdownState !== "idle"}
                />
                <div className="sensor-tooltip info">Total number of samples to be recorded</div>
              </div>

              <div className="sensor-tooltip-wrapper">
                <input
                  type="number"
                  id="duration"
                  className="input-box-sensor-one"
                  placeholder="Duration (in seconds)"
                  min="1"
                  value={inputValues.duration}
                  onWheel={(event) => event.currentTarget.blur()}
                  onChange={handleInputChange}
                  onBlur={handleInputBlur}
                  onKeyDown={(e) => handleKeyDown(e, "select_device")}
                  disabled={countdownState !== "idle"}
                />
                <div className="sensor-tooltip info">Duration of each sample in seconds</div>
              </div>

            </div>

            <div className="sensor-tooltip-wrapper">
              <div className="select-wrapper-sensor">
                <select
                  id="select_device"
                  className="input-box-sensor"
                  value={inputValues.select_device}
                  onChange={handleInputChange}
                  onKeyDown={(e) => handleKeyDown(e, "select_g")}
                  style={{ pointerEvents: countdownState !== "idle" ? "none" : "auto" }}
                >
                  <option value="IMU - MPU6050">IMU - MPU6050</option>
                  <option value="Temperature" disabled>Temperature (Unavailable)</option>
                  <option value="Humidity" disabled>Humidity (Unavailable)</option>
                </select>
              </div>
              <div className="sensor-tooltip info">Select the sensor to record data from</div>
            </div>


            <div className="sensor-tooltip-wrapper">
              <div className="select-wrapper-sensor">
                <select
                  id="select_g"
                  className="input-box-sensor"
                  value={inputValues.select_g}
                  onChange={handleInputChange}
                  onKeyDown={(e) => handleKeyDown(e, "person_name")}
                  style={{ pointerEvents: countdownState !== "idle" ? "none" : "auto" }}
                >
                  <option value="2g / 250° per sec">2g / 250° per sec</option>
                  <option value="4g / 500° per sec">4g / 500° per sec</option>
                  <option value="8g / 1000° per sec">8g / 1000° per sec</option>
                  <option value="16g / 2000° per sec">16g / 2000° per sec</option>
                </select>
              </div>
              <div className="sensor-tooltip info">Select sensor configuration</div>
            </div>


            <Button
              text={buttonText}
              className="start-graph-button"
              onClick={handleStartStop}
              disabled={buttonText === "Running..."}
            />

          </form>
        </div>
      </div>
    </div>
  );
};

export default SensorDataCollection;
import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import Button from "../components/Button";
import PageTransition from "../components/PageTransition";
import io from "socket.io-client";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper"
import CustomDropdown from "../components/CustomDropdown";
// Page shell lives in the .vdc-root section of this sheet - nothing here
// styles any other page, and the Audio page has its own sheet.
import "../styles/VideoDataCollection.css";
 
 
const VideoDataCollection = () => {
  // UI state management
  const [buttonText, setButtonText] = useState("Start Streaming");
  const [isStreaming, setIsStreaming] = useState(false);
  const [message, setMessage] = useState("");
  const [flashEffect, setFlashEffect] = useState(false);
 
  // Data and connection management
  const [inputValues, setInputValues] = useState({
    person_name: "",
    word_name: "",
    select_device: "GPX-Camera",
  });
  const [socket, setSocket] = useState(null);
  const [imageStream, setImageStream] = useState(null);
  const [captureCounts, setCaptureCounts] = useState({});
 
  // DOM references
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const currentRef = useRef(null);
 
  /**
   * Setup and cleanup socket connection on component mount/unmount
   */
  useEffect(() => {
    return () => {
      if (socket) {
        console.log("Disconnecting socket on unmount...");
        socket.emit("stop_image");
        socket.off("receive_image");
        socket.disconnect();
      }
    };
  }, [socket]);
 
 
  /**
   * Handles changes to input fields
   *
   * @param {Object} event - Event object from input change
   */
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
 
  /**
   * Toggles the video streaming state
   * Establishes or terminates socket connections for GPX camera
   */
  const toggleStreaming = async () => {
    const { person_name, word_name, select_device } = inputValues;
 
    if (!isStreaming) {
      // Input validation before starting stream
      if (!person_name || !word_name) {
        toast.warning("Please fill in all inputs before starting the stream.");
        return;
      }
 
      // Initialize socket connection for GPX camera
      if (select_device === "GPX-Camera" && !socket) {
        const socketConnection = io("http://127.0.0.1:9262"); // explicitly set backend URL
        socketConnection.on("connect", () => {
          console.log("Socket connected, emitting send_image");
          socketConnection.emit("send_image");
        });
        setSocket(socketConnection);
      }
 
 
      setIsStreaming(true);
      setButtonText("Stop Streaming");
    } else {
      // Stop and reset everything
      if (socket) {
        socket.emit("stop_image");
        socket.disconnect();
        setSocket(null);
      }
 
      setIsStreaming(false);
      setButtonText("Start Streaming");
 
      // **Reset all state values**
      setInputValues({
        person_name: "",
        word_name: "",
        select_device: "GPX-Camera",
      });
      setImageStream(null);
      setMessage("");
      setCaptureCounts({});
      setFlashEffect(false);
 
      // Reset webcam reference
      if (webcamRef.current) {
        webcamRef.current.srcObject = null;
      }
 
      if (currentRef.current) {
        currentRef.current.src = "";
      }
    }
  };
 
  /**
   * Socket event handler for receiving image data from GPX camera
   * Processes raw image data and converts it to displayable format
   */
  useEffect(() => {
  if (socket) {
    socket.on("error_update", (data) => {
  toast.error(data.message || "Hardware error");
 
 
  socket.emit("stop_image");
  socket.disconnect();
  setSocket(null);
 
  setIsStreaming(false);
  setButtonText("Start Streaming");
});
  }
 
  return () => {
    if (socket) {
      socket.off("error_update");
    }
  };
}, [socket]);
 
  useEffect(() => {
    if (socket) {
      socket.on("receive_image", (data) => {
        // Process incoming image data
        if (data && data.image && data.width && data.height) {
          const imgArray = data.image;
          const width = data.width;
          const height = data.height;
 
          // Create canvas to process grayscale image data
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
 
          const imageData = ctx.createImageData(width, height);
          const dataArray = imageData.data;
 
          // Convert single-channel grayscale to RGBA
          for (let i = 0; i < imgArray.length; i++) {
            const value = imgArray[i];
            dataArray[i * 4] = value;     // R
            dataArray[i * 4 + 1] = value; // G
            dataArray[i * 4 + 2] = value; // B
            dataArray[i * 4 + 3] = 255;   // Alpha (fully opaque)
          }
 
          ctx.putImageData(imageData, 0, 0);
          if (isStreaming) {
            setImageStream(canvas.toDataURL());
          } else {
            setImageStream(null);
          }
        } else {
          toast.error("Invalid image data received from the server.");
        }
      });
    }
 
    // Cleanup socket event listener
    return () => {
      if (socket) {
        socket.off("receive_image");
      }
    };
  }, [socket, isStreaming]);
 
  /**
   * Captures current frame from active camera source and saves it
   * Handles naming conventions, upload to server, and visual feedback
   */
  const captureFrame = async () => {
    // Validation checks
    if (!isStreaming) {
      toast.error("Please start streaming first.");
      return;
    }
 
    const { person_name, word_name, select_device } = inputValues;
 
    if (!person_name || !word_name) {
      toast.error("Please fill in both Person Name and Class Name fields.");
      return;
    }
 
    // Canvas setup for frame capture
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    let source;
 
    // Select appropriate video source based on device
    if (select_device === "GPX-Camera") {
      source = currentRef.current;
      canvas.width = 324;
      canvas.height = 324;
    } else {
      source = webcamRef.current.video;
      canvas.width = source.videoWidth;
      canvas.height = source.videoHeight;
    }
 
    // Draw frame to canvas and convert to base64
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/png");
 
    // Track capture counts for naming convention
    const key = `${person_name}_${word_name}_${select_device}`;
    let counter = captureCounts[key];
 
    if (!counter) {
      counter = 1;
    } else {
      counter += 1;
    }
 
    setCaptureCounts((prevCounts) => ({
      ...prevCounts,
      [key]: counter,
    }));
 
    // Determine the camera prefix for filename
    let cameraPrefix = select_device === "GPX-Camera" ? "GC" : "LC";
    const fileName = `${person_name}_${word_name}_${cameraPrefix}_${counter}.png`;
 
    // Upload captured frame to server
    try {
      const response = await fetch("http://127.0.0.1:9262/upload-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fileName,
          imageData,
          person_name,
          word_name,
          select_device,
        }),
      });
 
      if (response.ok) {
        setMessage(`Image saved in Collected_Data/Images/${fileName}`);
      } else {
        const errorData = await response.json();
        setMessage(`Error uploading image: ${errorData.message}`);
      }
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    }
 
    // Visual feedback for capture - flash effect
    setFlashEffect(true);
    setTimeout(() => setFlashEffect(false), 300);
  };
 
  return (
    <div className="vdc-root">
      <ToastWrapper />

      <div className="vdc-shell">
        {/* Navigation buttons for different data collection modes */}
        <aside className="vdc-rail">
          <span className="vdc-rail-mark"></span>

          <div className="vdc-nav-item">
            <div className="nav-tooltip-wrapper vdc-tile">
              <PageTransition to="/audiodatacollection">
                <img
                  src="assets/audio.png"
                  alt="Audio Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
            </div>
            <span className="vdc-nav-label">Audio</span>
          </div>

          <div className="vdc-nav-item is-current">
            <div className="nav-tooltip-wrapper vdc-tile is-current">
              <PageTransition to="/videodatacollection">
                <img
                  src="assets/vision.png"
                  alt="Video Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
            </div>
            <span className="vdc-nav-label">Vision</span>
          </div>

          {/* <div className="nav-tooltip-wrapper">
              <PageTransition to="/sensordatacollection">
                <img
                  src="assets/sensor_fusion.png"
                  alt="Sensor Data Collection"
                  className="button-icon"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              </PageTransition>
              <div className="nav-tooltip info">Sensor</div>
            </div> */}

          {/* <div className="vdc-nav-item is-locked">
            <div
              className="nav-tooltip-wrapper vdc-tile is-locked disabled-tab"
              onClick={() => toast.info("Sensor Data Collection is under development")}
            >
              <img
                src="assets/sensor_fusion.png"
                alt="Sensor Data Collection"
                className="sensor-button-icon"
                draggable="false"
                onContextMenu={(e) => e.preventDefault()}
              />
              <div className="nav-tooltip info">
                Sensor<br />(Under Development)
              </div>
            </div>
            <span className="vdc-nav-label">Sensor</span>
          </div> */}
        </aside>

        {/* Video preview and capture area */}
        <section className="vdc-stage">
          <div className="vdc-head">
            <div>
              <p className="vdc-eyebrow">Data Collection</p>
              <h2 className="vdc-title">Vision Capture Studio</h2>
              <p className="vdc-sub">
                Live camera preview and frame capture for the current class.
              </p>
            </div>
            <span className={`vdc-chip${isStreaming ? " is-live" : ""}`}>
              <span className="vdc-dot"></span>
              {isStreaming ? "Streaming Live" : "Standby"}
            </span>
          </div>

          <div className="vdc-frame-zone">
            <div
              className={`Streaming-video ${flashEffect ? "flash-effect" : ""}`}
            >
              {isStreaming && inputValues.select_device === "Laptop" ? (
                <Webcam
                  ref={webcamRef}
                  className="video"
                  audio={false}
                  screenshotFormat="image/png"
                />
              ) : inputValues.select_device === "GPX-Camera" ? (
                imageStream ? (
                  <img
                    ref={currentRef}
                    src={imageStream}
                    alt="Streaming Video"
                    className="vdc-stream-img"
                    draggable="false"
                    onContextMenu={(e) => e.preventDefault()}
                  />
                ) : (
                  <img
                    src="assets/disable_camera.png"
                    alt="Default"
                    className="default-image"
                    draggable="false"
                    onContextMenu={(e) => e.preventDefault()}
                  />
                )
              ) : (
                <img
                  src="assets/disable_camera.png"
                  alt="Default"
                  className="default-image"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              )}
            </div>
            {/* Capture button and status message */}
            <div className="vdc-capture">
              <Button
                text=""
                onClick={captureFrame}
                addClass="capture-button"
              />
              <p className="capture-message">{message}</p>
            </div>
          </div>

          <div className="vdc-meta">
            <div className="vdc-meta-item">
              <p className="vdc-meta-k">Camera</p>
              <p className="vdc-meta-v">{inputValues.select_device || "—"}</p>
            </div>
            <div className="vdc-meta-item">
              <p className="vdc-meta-k">Frame</p>
              <p className="vdc-meta-v">
                {inputValues.select_device === "GPX-Camera"
                  ? "324 × 324"
                  : "Native"}
              </p>
            </div>
            <div className="vdc-meta-item">
              <p className="vdc-meta-k">Format</p>
              <p className="vdc-meta-v">PNG</p>
            </div>
            <div className="vdc-meta-item">
              <p className="vdc-meta-k">Captures</p>
              <p className="vdc-meta-v">
                {Object.values(captureCounts).reduce((sum, n) => sum + n, 0)}
              </p>
            </div>
          </div>
        </section>

        {/* Form section for input parameters */}
        <section className="vdc-panel">
          <div className="vdc-panel-head">
            <p className="vdc-eyebrow">Session Setup</p>
            <h2 className="vdc-title">Collect Video Sample from Camera</h2>
            <p className="vdc-sub">
              Name the subject and class, pick a camera, then stream and capture
              frames.
            </p>
          </div>

          <form className="vdc-form">
            <div className="vdc-field">
              <label className="vdc-label" htmlFor="person_name">
                Person Name
              </label>
              <input
                type="text"
                spellCheck="false"
                id="person_name"
                className="vdc-input"
                placeholder="e.g. Alex"
                value={inputValues.person_name}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "word_name")}
              />
              <div className="vdc-tip">Enter the name of the person whose data is being recorded</div>
            </div>

            <div className="vdc-field is-right">
              <label className="vdc-label" htmlFor="word_name">
                Class Name
              </label>
              <input
                type="text"
                id="word_name"
                spellCheck="false"
                className="vdc-input"
                placeholder="e.g. thumbs_up"
                value={inputValues.word_name}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "select_device")}
              />
              <div className="vdc-tip">Enter the class or label name for this recording</div>
            </div>

            <div className="vdc-field is-full">
              <span className="vdc-label">Camera Input</span>
              {/* id + tabIndex keep the Enter-to-next-field chain landing here */}
              <div id="select_device" tabIndex={-1} className="vdc-dd-slot">
                <CustomDropdown
                  value={inputValues.select_device}
                  onChange={(val) => {
                    if (!isStreaming) {
                      setInputValues((prevValues) => ({
                        ...prevValues,
                        select_device: val,
                      }));
                    } else {
                      toast.error("Stop streaming to change cam");
                    }
                  }}
                  options={[
                    { value: "GPX-Camera", label: "GPX-Camera" },
                    { value: "Laptop", label: "Laptop Camera" },
                  ]}
                />
              </div>
              <div className="vdc-tip">Select the camera input</div>
            </div>
          </form>

          <div className="vdc-steps">
            <p className="vdc-steps-title">How it works</p>
            <ul>
              <li>
                <span>1</span>Fill in the person and class name, then choose a
                camera.
              </li>
              <li>
                <span>2</span>Start streaming to bring up the live preview.
              </li>
              <li>
                <span>3</span>Press the shutter for every frame you want saved
                to Collected_Data/Images.
              </li>
            </ul>
          </div>

          {/* Main control button */}
          <div className="vdc-actions">
            <Button
              text={buttonText}
              className="recording-button"
              onClick={toggleStreaming}
            />

          </div>
        </section>
      </div>
      {/* Hidden canvas for frame processing */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
};
 
export default VideoDataCollection;
 
 
 

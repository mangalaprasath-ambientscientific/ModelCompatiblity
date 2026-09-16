// ===============================
// Debugger.jsx
// ===============================
import React, { useState, useRef, useEffect } from "react";
import { io } from "socket.io-client";
import "../styles/DebuggerPage.css";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper";

//Main Debugger component

const Debugger = () => {

  const [compilationFailed, setCompilationFailed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineStarted, setPipelineStarted] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);
  const [selectedConfigFile, setSelectedConfigFile] = useState(null);
  const [selectedInputFile, setSelectedInputFile] = useState(null);
  const [compatibilityResult, setCompatibilityResult] = useState(null);
  const [showLayers, setShowLayers] = useState(false);
  const [activeLoadingIndex, setActiveLoadingIndex] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [progressLogs, setProgressLogs] = useState([]);
  const [showSuccess, setShowSuccess] = useState(false);
  const [failureMessage, setFailureMessage] = useState("");
  const [draggingType, setDraggingType] = useState(null);
  const configFileRef = useRef();
  const inputFileRef = useRef();

  /*** ===============================
   * Helper: File status summary
   * ===============================*/
  const getFileStatus = () => {
    const modelLoaded = !!selectedConfigFile;
    const inputLoaded = !!selectedInputFile;
    if (modelLoaded && inputLoaded) {
      const modelExt = selectedConfigFile.name.split(".").pop().toLowerCase();
      return `.${modelExt} file and input file loaded`;
    } else if (modelLoaded) {
      const modelExt = selectedConfigFile.name.split(".").pop().toLowerCase();
      return `.${modelExt} file loaded`;
    } else if (inputLoaded) {
      return `Input file loaded`;
    } else {
      return "No file loaded";
    }
  };
  // Drag & Drop Handlers

  const handleDragOver = (e, type) => {
    if (isLoading || pipelineStarted) return;   
    e.preventDefault();
    setDraggingType(type);
  };

  const handleDragLeave = () => {
    if (isLoading || pipelineStarted) return;   
    setDraggingType(null);
  };

  const handleDrop = (e, type) => {
    if (isLoading || pipelineStarted) return;  
    e.preventDefault();
    setDraggingType(null);

    const file = e.dataTransfer.files[0];
    if (!file) return;

    const ext = file.name.split(".").pop().toLowerCase();

    if (type === "config" && (ext === "h5" || ext === "tflite")) {
      setSelectedConfigFile(file);
    } else if (type === "input" && (ext === "txt" || ext === "npy")) {
      setSelectedInputFile(file);
    } else {
      toast.warning("Unsupported file type for this field.");
    }
  };

  // ===============================
  // Socket.IO Connection & Handlers
  // ===============================
  useEffect(() => {
    if (isLoading || pipelineStarted) {
      window.__DEBUGGER_RUNNING__ = true;
    } else {
      window.__DEBUGGER_RUNNING__ = false;
    }
  }, [isLoading, pipelineStarted]);

  useEffect(() => {
    const socket = io("http://localhost:9262");
    // Handle progress updates
    socket.on("progress_update", (data) => {
      const prog = data.progress ?? 0;
      const msg = data.message || "";

      setProgress(prog);
      setProgressMessage(msg);

      if (msg) {
        setProgressLogs((prev) => {
          const newLogs = [...prev];

          // Reset previous loader when a new log comes
          if (
            activeLoadingIndex !== null &&
            activeLoadingIndex < newLogs.length &&
            !msg.toLowerCase().includes("extracting layer-wise outputs") &&
            !msg.toLowerCase().includes("compilation in progress") &&
            !msg.toLowerCase().includes("generating excel sheet")
          ) {
            setActiveLoadingIndex(null);
          }

          newLogs.push(msg);

          // Detect long-running logs
          if (
            msg.toLowerCase().includes("extracting layer-wise outputs") ||
            msg.toLowerCase().includes("compilation in progress") ||
            msg.toLowerCase().includes("generating excel sheet")

          ) {
            setActiveLoadingIndex(newLogs.length - 1);
          }
          return newLogs;
        });
      }
      if (
        prog === 100 &&
        msg.toLowerCase().includes("excel generated successfully")
      ) {
        setProgressMessage("Excel generation completed — finalizing...");

        window.__DEBUGGER_SUCCESS__ = true;
        window.__DEBUGGER_RUNNING__ = false;

        setTimeout(() => {
          setShowSuccess(true);
          setIsLoading(false);
        }, 3000);
      }

      if (msg.toLowerCase().includes("excel sheet generated successfully")) {

        window.__DEBUGGER_SUCCESS__ = true;
        window.__DEBUGGER_RUNNING__ = false;

        setTimeout(() => {
          setShowSuccess(true);
          setIsLoading(false);
        }, 3000);
      }
    });

    // Handle errors
    socket.on("error_update", async (data) => {
      const stage = data?.stage || "";
      let finalMessage = data?.message || "Unknown error occurred.";
      setFailureMessage(finalMessage);
      // MODEL VALIDATION ONLY
      if (stage === "model_validation") {
        setIsLoading(false);
        setPipelineStarted(false);
        setShowLayers(false);
        setCompatibilityResult(null);
        await new Promise((resolve) => setTimeout(resolve, 3500));
        return;
      }
      if (data?.message?.toLowerCase().includes("ram overflow")) {
        data.message = "RAM Overflow: Model/input exceeds available RAM";
      }

      // INPUT VALIDATION
      if (stage === "input_validation" && pipelineStarted) {
        toast.error("Invalid input file. Please check your input content.");
        setIsLoading(false);
        setPipelineStarted(false);
        setProgress(0);
        return;
      }

      // DEFAULT HARDWARE / COMPILATION FAILURE
      toast.error(finalMessage);
      setIsLoading(false);
      setPipelineStarted(false);
      setCompatibilityResult(null);
      setHasStarted(false);
      setProgress(0);
      setProgressMessage("");
      setProgressLogs([]);
      setShowLayers(false);
      setShowSuccess(false);
      setActiveLoadingIndex(null);
      setCompilationFailed(true);
    });

    return () => socket.disconnect();
  }, [pipelineStarted, activeLoadingIndex]);

  // ===============================
  // Handlers: Pipeline & File Upload
  // ===============================
  const handleStart = async () => {

    window.__DEBUGGER_SUCCESS__ = false;
    window.__DEBUGGER_RUNNING__ = true;

    if (!selectedConfigFile || !selectedInputFile) {
      toast.warning("Please upload model and input files.");
      return;
    }
    setCompilationFailed(false);
    setHasStarted(true);
    setIsLoading(true);
    setCompatibilityResult(null);
    setProgressLogs([]);
    setProgress(0);
    setProgressMessage("Checking model compatibility...");
    setShowLayers(false);
    setPipelineStarted(false);
    const formData = new FormData();
    formData.append("model_file", selectedConfigFile);
    formData.append("input_file", selectedInputFile);
    try {
      const response = await fetch("http://localhost:9262/run-model", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (
        response.status === 400 &&
        result?.error?.includes("Hardware connection missing")
      ) {
        setIsLoading(false);
        return;
      }

      if (!response.ok) {
        setIsLoading(false);
        return;
      }

      setCompatibilityResult(result);
      setShowLayers(true);
      setCompilationFailed(false);

      if (result.status === "incompatible") {
        if (
          (!result.layer_details || result.layer_details.length === 0) &&
          result.suggestions && result.suggestions.length > 0
        ) {
          toast.error(result.suggestions[0]);      //popup
          setFailureMessage(result.suggestions[0]); // message text
          setShowLayers(false);
          setCompilationFailed(true);   // THIS ENABLES ERROR UI
          //  no layers
        }
        // LAYER-LEVEL ERRORS
        else {
          toast.error("Model is incompatible for GPX – Please check");
          setFailureMessage("Model is incompatible for GPX – Please check");
          setShowLayers(true);
        }
        setIsLoading(false);
        setPipelineStarted(false);
        setProgress(0);
        return;
      }


      if (result.status === "incompatible_input") {
        toast.error("Input file is invalid – Please check");
        setShowLayers(true);
        setIsLoading(false);
        setPipelineStarted(false);
        setCompilationFailed(false);
        setProgress(0)

        return;
      }
      // If compatible, proceed to pipeline execution
      setTimeout(() => {
        setShowLayers(false);
        setProgressLogs(["Compatibility check passed"]);
        setProgress(10);

        setPipelineStarted(true);
        handleStartPipeline();
      }, 9000);
    } catch (err) {
      console.error(err);
      toast.error("Could not reach server");
      setIsLoading(false);
    }
  };
  // /Trigger backend execution
  const handleStartPipeline = async () => {
    try {
      await fetch("http://localhost:9262/start-pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: selectedConfigFile.name,
          input_path: selectedInputFile.name,
        }),
      });
    } catch (err) {
      console.error(err);
      toast.error("Failed to start pipeline.");
      setPipelineStarted(false);
      setIsLoading(false);
    }
  };

  const handleConfigFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setSelectedConfigFile(file);
    setHasStarted(false);
    setCompatibilityResult(null);
    setShowLayers(false);
    setPipelineStarted(false);
  };

  const handleInputFileChange = (e) => {
    const file = e.target.files[0];
    if (file) setSelectedInputFile(file);
  };

  const handleBrowseClick = (type) => {
    if (type === "config") configFileRef.current.click();
    if (type === "input") inputFileRef.current.click();
  };
  // ===============================
  // Reset Function
  // ===============================
  const resetDebugger = () => {
    setSelectedConfigFile(null);
    setSelectedInputFile(null);
    window.__DEBUGGER_SUCCESS__ = false;
    window.__DEBUGGER_RUNNING__ = false;
    setIsLoading(false);
    setPipelineStarted(false);
    setCompatibilityResult(null);
    setHasStarted(false);
    setProgress(0);
    setProgressMessage("");
    setProgressLogs([]);
    setShowLayers(false);

    if (configFileRef.current) configFileRef.current.value = "";
    if (inputFileRef.current) inputFileRef.current.value = "";
  };
  useEffect(() => {
    const handleFlasherReset = () => resetDebugger();
    window.addEventListener("flasher-reset", handleFlasherReset);

    return () => window.removeEventListener("flasher-reset", handleFlasherReset);
  }, []);

  // ===============================
  // JSX Render
  // ===============================
  return (
    <>
      <div className={`debugger-container-wrapper ${showSuccess ? "debugger-blurred" : ""}`}>
        <div className={`debugger-container ${hasStarted ? "debugger-shift-down" : ""}`}>
          <ToastWrapper />

          {/* LEFT SIDE */}
          <div className="debugger-leftside">
            <span className="debugger-progress-percentage">{progress}%</span>

            <div className="debugger-progress-bar">
              <div
                className="debugger-progress-fill"
                style={{
                  width: `${progress}%`,
                  backgroundColor: "#53d824",
                  height: "100%",
                }}
              />
            </div>
            <div className="debugger-leftbox">
              <div className="debugger-left-box-title">
                <span className="debugger-left-box-title-text">
                  {pipelineStarted ? "Getting Layerwise Outputs" : "Compatibility Check"}
                </span>
                <span className="debugger-status-text">
                  Status:{" "}
                  {compatibilityResult?.status === "incompatible" ? (
                    <span style={{ color: "#ff4d4d" }}>Model Incompatible</span>
                  ) : progress >= 100 ? (
                    "Output Completed"
                  ) : isLoading ? (
                    pipelineStarted ? "Processing layerwise..." : "Checking..."
                  ) : compatibilityResult?.status === "compatible" ? (
                    "Compatibility Passed"
                  ) : compatibilityResult?.status === "incompatible_input" ? (
                    "Input File Invalid"
                  ) : (
                    getFileStatus()
                  )}
                </span>
              </div>
              <div className="debugger-left-box-content">

                {compilationFailed && (
                  <div className="compilation-failed-container">
                    <img src="./assets/failure.png" className="failure-image" alt="failed" />
                    <div className="compilation-failed-text">
                      {failureMessage || "Execution Failed"}
                    </div>
                  </div>
                )}
                {showLayers &&
                  compatibilityResult && (
                    <div className="debugger-compatibility-results">

                      {compatibilityResult.layer_details.map((layer, idx) => {
                        const entry = Array.isArray(layer[0]) ? layer[0] : layer;

                        const layerTypeRaw = entry[1] || "Unknown";
                        const layerType = layerTypeRaw.replace(/\.+$/, "");
                        const status = entry[2] || "";
                        const extra = entry[3] || "";
                        const normalizedStatus = status.toLowerCase();
                        const normalizedExtra = extra.toLowerCase();

                        const isExplicitLayerError =
                          normalizedExtra.includes(`'${layerType.toLowerCase()}' not supported`) ||
                          normalizedExtra === `${layerType.toLowerCase()} not supported`;

                        // Layer-level failure
                        const isLayerNotSupported =
                          isExplicitLayerError ||
                          (normalizedStatus.includes("not supported") &&
                            !normalizedStatus.includes("operator"));

                        // Operator-level failure
                        const isOperatorNotSupported =
                          !isLayerNotSupported &&
                          normalizedStatus.includes("operator not supported");
                        const isSupported =
                          !isLayerNotSupported && !isOperatorNotSupported;
                        const reason = !isSupported && extra ? extra : "";
                        return (
                          <div
                            key={idx}
                            className="debugger-layer-item debugger-fade-in"
                            style={{ animationDelay: `${idx * 0.25}s` }}
                          >
                            <div
                              className="debugger-layer-line1"
                              style={{ color: isSupported ? "#ececec" : "#ff4d4d" }}
                            >
                              Layer {idx + 1}
                            </div>

                            <div className="debugger-layer-line2">
                              <span className="layer-type" style={{ color: "#ececec" }}>
                                {layerType}
                              </span>
                              <span
                                className="layer-status"
                                style={{ color: isSupported ? "#53d824" : "#ff4d4d" }}
                              >
                                {isSupported
                                  ? "Layer Supported ✔"
                                  : isLayerNotSupported
                                    ? "Layer Not Supported ✖"
                                    : "Operator Not Supported ✖"}
                              </span>
                            </div>
                            {!isSupported && reason && (
                              <div
                                className="debugger-layer-reason"
                                style={{ color: "#ff4d4d" }}
                              >
                                {reason}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}

                {/* 3️ Default — progress logs */}
                {!compatibilityResult || (compatibilityResult.status === "compatible" && !showLayers) ? (
                  <div className="debugger-progress-logs">
                    {progressLogs.map((log, idx) => {
                      const isLoadingLog = idx === activeLoadingIndex;
                      return (
                        <div
                          key={idx}
                          className="debugger-progress-log-item debugger-fade-in"
                          style={{ animationDelay: `${idx * 0.2}s` }}
                        >
                          {/* log text */}
                          <span className="log-text">{log}</span>

                          {/* loader or tick */}
                          {isLoadingLog ? (
                            <span className="inline-loader log-status">
                              <div className="dot"></div>
                              <div className="dot"></div>
                              <div className="dot"></div>
                            </span>
                          ) : (
                            <span className="log-tick log-status">✔</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : null}

              </div>
            </div>
          </div>

          {/* RIGHT SIDE */}
          <div className="debugger-divider" />
          <div className="debugger-rightside">
            <div className="debugger-title">
              <h1>Debugger</h1>
              <div className="debugger-titleline"></div>
            </div>

            {/* Input Section */}
            <div className="debugger-inputs">
              {/* === MODEL UPLOAD === */}
              <span className="debugger-labels">Model:</span>
              <div
                className={`debugger-portion1 drop-zone ${draggingType === "config" ? "drag-active" : ""
                  }`}
                onDragOver={(e) => handleDragOver(e, "config")}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, "config")}
              >
                <div className="debugger-input-section tooltip-wrapper">
                  <span className="debugger-file-name">
                    {draggingType === "config"
                      ? "Drop .h5 or .tflite file here"
                      : selectedConfigFile
                        ? selectedConfigFile.name
                        : "Upload Model File"}
                  </span>
                  {/* MODEL FILE INPUT */}
                  <input
                    type="file"
                    accept=".h5"
                    ref={configFileRef}
                    onChange={handleConfigFileChange}
                    className="debugger-file-input"
                  />
                  <span className="tooltip debugger-info">
                    Upload the model file (.h5).

                  </span>
                </div>
                {/* MODEL BROWSE BUTTON */}
                <button
                  className={`debugger-browse-button ${isLoading ? "debugger-disabled" : ""}`}
                  onClick={() => !isLoading && handleBrowseClick("config")}
                  disabled={isLoading}
                  style={{
                    cursor: isLoading ? "not-allowed" : "pointer",
                  }}
                >
                  Browse...
                </button>
              </div>

              {/* === INPUT UPLOAD === */}
              <span className="debugger-labels">Input:</span>
              <div
                className={`debugger-portion1 drop-zone ${draggingType === "input" ? "drag-active" : ""
                  }`}
                onDragOver={(e) => handleDragOver(e, "input")}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, "input")}
              >
                <div className="debugger-input-section tooltip-wrapper">
                  <span className="debugger-file-name">
                    {draggingType === "input"
                      ? "Drop .txt or .npy file here"
                      : selectedInputFile
                        ? selectedInputFile.name
                        : "Upload Input Text File"}
                  </span>
                  {/* INPUT FILE INPUT */}
                  <input
                    type="file"
                    accept=".txt,.npy"
                    ref={inputFileRef}
                    onChange={handleInputFileChange}
                    className="debugger-file-input"
                  />
                  <span className="tooltip debugger-info">
                    Upload the input file (.txt).

                  </span>
                </div>
                {/* INPUT BROWSE BUTTON */}
                <button
                  className={`debugger-browse-button ${isLoading ? "debugger-disabled" : ""}`}
                  onClick={() => !isLoading && handleBrowseClick("input")}
                  disabled={isLoading}
                  style={{
                    cursor: isLoading ? "not-allowed" : "pointer",
                  }}
                >
                  Browse...
                </button>

              </div>
              {/* Action Button */}
              <button
                className={`debugger-start-button ${isLoading ? "debugger-disabled" : ""
                  }`}
                onClick={!isLoading ? handleStart : undefined}
                disabled={isLoading}
                style={{
                  cursor: isLoading ? "not-allowed" : "pointer",
                  transition: "all 0.3s ease",
                }}
              >
                {isLoading ? "Processing..." : "Get Layerwise Output"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Success Popup */}
      {showSuccess && (
        <div className="debugger-popup-overlay">
          <div className="debugger-popup-box">
            <div className="debugger-popup-tick">✔</div>
            <h2>Layer-wise Output Generated Successfully!</h2>
            <p>Check the Result Log page for details.</p>
            <button
              className="debugger-popup-button"
              onClick={async () => {
                try {
                  await fetch("http://localhost:9262/reset-debugger", {
                    method: "POST",
                  });
                  console.log("Reset signal sent");
                } catch (err) {
                  console.error("Failed to send reset:", err);
                }
                window.location.reload();
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Debugger;


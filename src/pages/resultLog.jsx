import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper"
import { confirmAlert } from "react-confirm-alert";
import "react-confirm-alert/src/react-confirm-alert.css";
 
const ResultLog = () => {
  const [sliderValue, setSliderValue] = useState(1);
  const sliderRef = useRef(null);
  const sliderValues = [" ", "Small", "Medium", "Large"];
  const dotsInterval = useRef(null);
  const [animatedDots, setAnimatedDots] = useState("");
  const [files, setFiles] = useState([]);
  const [runningState, setRunningState] = useState({});
  const [selectedOption, setSelectedOption] = useState("Label");
  const [isRunning, setIsRunning] = useState(false);
  axios.defaults.baseURL = "http://localhost:9262"; // or your Flask backend URL
 
  useEffect(() => {
    axios
      .get("/api/files")
      .then((response) => {
        setFiles(response.data);
      })
      .catch((error) => {
        console.error("Error fetching files:", error);
      });
 
    const storedRunningState = JSON.parse(localStorage.getItem("runningState"));
    if (storedRunningState) {
      setRunningState(storedRunningState);
    }
  }, []);
 
  const handleOpenFile = (fileName) => {
    const link = document.createElement("a");
    link.href = `http://localhost:9262/api/files/${encodeURIComponent(fileName)}`;
    link.setAttribute("download", fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
 
 
  // const handleOpenFile = (fileName) => {
  //   window.open(`/api/files/${fileName}`, "_blank");
  // };
 
  const handleModelChange = (newSliderValue) => {
    if (isRunning) {
      toast.error("Please wait, Previous model is running");
      return;
    }
    setSliderValue(newSliderValue);
  };
 
  const handleDelete = (fileName) => {
    confirmAlert({
      customUI: ({ onClose }) => (
        <div
          style={{
            backgroundColor: "#151515",
            color: "#ECECEC",
            padding: "20px",
            borderRadius: "8px",
            fontFamily: "var(--es-sans)",
          }}
        >
          <h1 style={{ color: "#ECECEC" }}>Confirm to delete</h1>
          <p>
            Are you sure you want to delete the file: <br />
            <strong style={{ display: "block", marginTop: "10px" }}>
              {fileName}
            </strong>
          </p>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <button
              onClick={() => {
                axios
                  .delete(`/api/delete-file/${encodeURIComponent(fileName)}`)
                  .then((response) => {
                    toast.success("File deleted successfully!");
                    setFiles(files.filter((file) => file.name !== fileName));
                  })
                  .catch((error) => {
                    console.error(
                      "Error deleting file:",
                      error.response?.data || error.message
                    );
                    toast.error(
                      `Error: ${error.response?.data?.message ||
                      "Unable to delete file."
                      }`
                    );
                  });
                onClose();
              }}
              style={{
                backgroundColor: "#53D824",
                color: "#ECECEC",
                fontFamily: "var(--es-sans)",
                padding: "10px 20px",
                borderRadius: "4px",
                border: "none",
              }}
            >
              Yes
            </button>
            <button
              onClick={() => {
                onClose();
              }}
              style={{
                backgroundColor: "red",
                color: "#ECECEC",
                fontFamily: "var(--es-sans)",
                padding: "10px 20px",
                borderRadius: "4px",
                border: "none",
              }}
            >
              No
            </button>
          </div>
        </div>
      ),
    });
  };
 
  const startDotsAnimation = () => {
    let dots = "";
    dotsInterval.current = setInterval(() => {
      dots = dots.length < 3 ? dots + "." : "";
      setAnimatedDots(dots);
    }, 300);
  };
 
  const stopDotsAnimation = () => {
    clearInterval(dotsInterval.current);
    setAnimatedDots("");
  };
 
  const handleTranscribe = (fileName) => {
    if (isRunning) {
      toast.info("Previous transcription is not completed");
      return;
    }
    setIsRunning(true);
    setRunningState((prevState) => {
      const newState = { ...prevState, [fileName]: true };
      localStorage.setItem("runningState", JSON.stringify(newState));
      return newState;
    });
    startDotsAnimation();
 
    axios
      .post("/api/transcribe", {
        model_choice: sliderValue,
        file_name: fileName,
      })
      .then((response) => {
        toast.success("Transcription completed successfully!");
 
        axios
          .get("/api/files")
          .then((response) => {
            setFiles(response.data);
          })
          .catch((error) => {
            toast.error("Failed to fetch updated files.");
          });
      })
      .catch((error) => {
        console.error(
          "Error during transcription:",
          error.response?.data || error.message
        );
        toast.error("Error during transcription. Check console for details.");
      })
      .finally(() => {
        stopDotsAnimation();
        setRunningState((prevState) => {
          const newState = { ...prevState, [fileName]: false };
          localStorage.setItem("runningState", JSON.stringify(newState));
          return newState;
        });
        setIsRunning(false);
      });
  };
 
  const handleAddToDataset = (fileName) => {
    axios
      .post("/api/add_to_dataset", {
        file_name: fileName,
        selected_option: selectedOption,
      })
      .then((response) => {
        toast.success("File added to dataset successfully!");
      })
      .catch((error) => {
        console.error(
          "Error adding to dataset:",
          error.response?.data || error.message
        );
        toast.error("Error adding to dataset. Check console for details.");
      });
  };
 
  useEffect(() => {
    const eventSource = new EventSource("/api/updates");
 
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.update) {
        axios
          .get("/api/files")
          .then((response) => {
            setFiles(response.data);
          })
          .catch((error) => {
            toast.error("Failed to fetch updated files.");
          });
      }
    };
 
    return () => {
      eventSource.close();
    };
  }, []);
 
  useEffect(() => {
    const slider = sliderRef.current;
    if (slider) {
      const min = parseInt(slider.min, 10);
      const max = parseInt(slider.max, 10);
      const percentage = ((sliderValue - min) / (max - min)) * 100;
      slider.style.background = `linear-gradient(to right, #53d824 ${percentage}%, #ececec ${percentage}%)`;
    }
  }, [sliderValue]);
 
  return (
    <div>
      <ToastWrapper />
      <div className="container-screen">
        <div className="top-screen">
          <div className="select-model-result">
            <p>Select your model</p>
            <div className="slider-container-result">
              <input
                ref={sliderRef}
                type="range"
                className="slider-slider-result"
                min="1"
                max="3"
                step="1"
                value={sliderValue}
                onChange={(e) => handleModelChange(Number(e.target.value))}
              />
              <div className="slider-labels-result">
                <span>Small</span>
                <span>Medium</span>
                <span>Large</span>
              </div>
            </div>
          </div>
          <h1>Result Log</h1>
        </div>
 
        <div className="bottom-screen">
          <div className="sheets-container">
            <h1>Result Sheets</h1>
          </div>
          <div className="box">
            {files.length === 0 ? (
              <p>No files found</p>
            ) : (
              <ul>
                {files.map((file) => (
                  <li key={file.name} className="file-item">
                    <div className="file-details">
                      <p className="file-name">
                        <img
                          src="./assets/sheet_file.png"
                          alt="Excel Icon"
                          className="excel-icon"
                          draggable="false"
                          onContextMenu={(e) => e.preventDefault()}
                        />
                        <strong>{file.name}</strong>
                      </p>
                      <div className="file-meta">
                        <p className="file-modified">
                          Last Modified: {file.modified}
                        </p>
                        {runningState[file.name] && (
                          <p className="running-message">
                            {sliderValues[sliderValue]} model is running
                            {animatedDots}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="file-actions">
                      <button onClick={() => handleOpenFile(file.name)}>Download</button>
 
                      {file.name.startsWith("Collected_Data_Sensor") ? (
                        <>
                          <button
                            onClick={() => handleDelete(file.name)}
                            style={{ backgroundColor: "red", color: "#ECECEC" }}
                          >
                            Delete
                          </button>
                        </>
                      ) : file.name.startsWith("Collected_Data_Audio") ? (
                        <>
                          {/* Hide Transcribe if already transcribed */}
                          {!file.name.includes("Transcribed") && (
                           <button
  onClick={() => !runningState[file.name] && handleTranscribe(file.name)}
  disabled={runningState[file.name] || file.transcriptAvailable}
  className={
    (runningState[file.name] || file.transcriptAvailable)
      ? "disabled-button transcribing-btn"
      : ""
  }
>
  {runningState[file.name] ? "Transcribing..." : "Transcribe"}
</button>
 
 
                          )}
                          <button
                            onClick={() => {
                              if (runningState[file.name]) {
                                toast.warning("Transcription is running... Wait a while.");
                                return;
                              }
                              handleDelete(file.name);
                            }}
                            style={{ backgroundColor: "red", color: "#ECECEC" }}
                          >
                            Delete
                          </button>
                        </>
                      ) : (
                        <>
                          {/* Hide Transcribe for Layer-wise and Excel files */}
                          {!file.name.startsWith("Layer_Wise") &&
                            !file.name.endsWith(".xlsx") &&
                            !file.name.includes("_Transcribed") && (
                              <button
  onClick={() => !runningState[file.name] && handleTranscribe(file.name)}
  disabled={runningState[file.name] || file.transcriptAvailable}
  className={
    (runningState[file.name] || file.transcriptAvailable)
      ? "disabled-button transcribing-btn"
      : ""
  }
>
  {runningState[file.name] ? "Transcribing..." : "Transcribe"}
</button>
 
 
                            )}
 
                          {/* <button onClick={() => handleAddToDataset(file.name)}>
                            Add to Dataset
                          </button> */}
                          {/* <select
                            defaultValue="Label"
                            onChange={(e) => setSelectedOption(e.target.value)}
                          >
                            <option value="Label">Label</option>
                            {file.transcriptAvailable && (
                              <option value="Transcript">Transcript</option>
                            )}
                          </select> */}
 
                          <button
                            onClick={() => handleDelete(file.name)}
                            style={{ backgroundColor: "red", color: "#ECECEC" }}
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
 
export default ResultLog;
 
 
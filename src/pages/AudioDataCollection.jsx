import React, { useState, useRef } from "react";
import Button from "../components/Button";
import PageTransition from "../components/PageTransition";
import { toast } from "react-toastify";
import ToastWrapper from "../components/ToastWrapper";
import CustomDropdown from "../components/CustomDropdown";
import { io } from "socket.io-client";
import { useEffect } from "react";
import { API } from "../config";
// Page shell lives in the .adc-root section of this sheet - nothing here
// styles any other page, and the Vision page has its own sheet.
import "../styles/audioDataCollection.css";

const AudioDataCollection = () => {
  const [buttonText, setButtonText] = useState("Start Recording");
  const [animatedDots, setAnimatedDots] = useState("");
  const [referenceText, setReferenceText] = useState("");
  const [countdownText, setCoundownText] = useState(null);
  const progressCricle = useRef(null);
  const rootRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);

  const [inputValues, setInputValues] = useState({
    person_name: "",
    word_name: "",
    select_device: "GPX-I2S",
    duration_of_sample: "",
    number_of_samples: "",
    frequency: 8,
  });
  const dotsInterval = useRef(null);

  const handleInputChange = (e) => {
    const { id, value } = e.target;

    // Person Name & Word to Record
    if (id === "person_name" || id === "word_name") {
      if (value !== "" && !/^[a-zA-Z][a-zA-Z0-9_ ]*$/.test(value)) {
        return;
      }
    }

    // Recording Duration - allow only positive integers
    if (id === "duration_of_sample") {
      if (value !== "" && !/^\d+$/.test(value)) {
        return;
      }

      if (value !== "" && Number(value) < 1) {
        return;
      }
    }

    // Number of Samples - allow only positive integers
    if (id === "number_of_samples") {
      if (value !== "" && !/^\d+$/.test(value)) {
        return;
      }

      if (value !== "" && Number(value) < 1) {
        return;
      }
    }

    setInputValues((prevValues) => ({
      ...prevValues,
      [id]: value,
    }));
  };

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

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

  const startProgressCircle = async (duration_of_sample, number_of_samples) => {
    // Safely apply circle state; no-ops if the element is gone (unmounted mid-await)
    const setCircle = ({ rotating, duration } = {}) => {
      const el = progressCricle.current;
      if (!el) return false;              // element unmounted -> skip
      if (rotating === true) {
        el.classList.add("progress-circle-rotating");
        el.classList.remove("progress-circle-static");
      } else if (rotating === false) {
        el.classList.remove("progress-circle-rotating");
        el.classList.add("progress-circle-static");
      }
      if (duration !== undefined) el.style.animationDuration = duration;
      return true;
    };

    for (let i = 0; i < number_of_samples; i++) {
      setReferenceText("Get ready to speak...");
      await delay(2000);

      for (let countdown = 3; countdown > 0; countdown--) {
        setReferenceText(countdown.toString());

        if (countdown === 1) {
          try {
                    await fetch(`${API}/countdown-notification`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message: "start recording" }),
            });
            console.log("Message sent!");
          } catch (error) {
            console.error("Error sending countdown notification:", error);
          }
        }

        await delay(1000);
      }

      // start rotating; if the circle is gone, bail out of the whole routine
      if (!setCircle({ rotating: true, duration: `${duration_of_sample}s` })) return;

      // Visual feedback during recording
      setReferenceText("Recording");
      startDotsAnimation();
      setCoundownText(Number(duration_of_sample));
      const timerId = setInterval(() => {
        setCoundownText((prevTime) => {
          if (prevTime <= 0) {
            clearInterval(timerId);
            return 0;
          }
          return (prevTime - 0.01).toFixed(2);
        });
      }, 10);

      await delay(duration_of_sample * 1000);
      clearInterval(timerId);

      // Reset animation state between samples
      setCircle({ rotating: false, duration: "" });
      stopDotsAnimation();
      setReferenceText("Recording complete. Get ready for the next sample...");
    }

    // Final processing and completion
    setReferenceText("Processing data...");
    await delay(4000);
    try {
        await fetch(`${API}/trigger-end`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "recording complete" }),
      });
      console.log("End trigger sent!");
    } catch (error) {
      console.error("Error sending end trigger:", error);
    }
    setReferenceText("Consolidating data...");
    await delay(4000);

    setCircle({ rotating: false, duration: "" });

    await delay(100);
    toast.success("Data Collection Complete. Check Result Log");
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

  const handleButtonClick = async () => {
    const {
      person_name,
      word_name,
      select_device,
      duration_of_sample,
      number_of_samples,
      frequency,
    } = inputValues;

    if (!isRecording) {
      // Input validation
      if (
        !person_name ||
        !word_name ||
        !select_device ||
        !duration_of_sample ||
        !number_of_samples ||
        !frequency
      ) {
        toast.warning("Please fill all the input fields correctly!");
        return;
      }

      if (
        isNaN(Number(duration_of_sample)) ||
        isNaN(Number(number_of_samples)) ||
        isNaN(Number(frequency))
      ) {
        toast.error(
          "Please enter valid numeric values for duration, samples, and frequency."
        );
        return;
      }

      if (Number(duration_of_sample) <= 0 || Number(number_of_samples) <= 0) {
        toast.error(
          "Duration of sample and number of samples must be greater than 0."
        );
        return;
      }

      if (select_device !== "GPX-I2S") {
        toast.error("Please select 'GPX-I2S' to enable the laptop microphone.");
        return;
      }

      // Start recording process
      setButtonText("Fetching Text...");
      setIsRecording(true);

      try {
        // Prepare payload for backend
        const payload = {
          person_name,
          word_name,
          select_device,
          duration_of_sample: Number(duration_of_sample),
          num_samples: Number(number_of_samples),
          frequency: Number(frequency),
        };

        // Send request to backend API
                const response = await fetch(`${API}/process-audio`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          setIsRecording(false);
          setButtonText("Start Recording");
          return;

        }

        setReferenceText("Initializing...");

        // Poll for microphone initialization status
        const fetchTextPromise = new Promise(async (resolve, reject) => {
          let text = "";
          while (text.trim() !== "GPX I2S Mic Initialized") {
            try {
                            const textResponse = await fetch(`${API}/get-text-audio`);
              text = await textResponse.text();
              setReferenceText(text);

              if (text.trim() === "GPX I2S Mic Initialized") {
                resolve(text);
              } else {
                await delay(1000);
              }
            } catch (error) {
              reject(error);
            }
          }
        });

        // Once mic is initialized, start recording process
        fetchTextPromise.then(async () => {
          setButtonText("Running...");
          await startProgressCircle(duration_of_sample, number_of_samples);
          setButtonText("Start Recording");
          setIsRecording(false);

          // Reset form after completion
          setInputValues({
            person_name: "",
            word_name: "",
            select_device: "GPX-I2S",
            duration_of_sample: "",
            number_of_samples: "",
            frequency: 8,
          });
          setReferenceText("");
          setCoundownText(null);
        });


      } catch (error) {
        toast.error(
          "An error occurred while sending data or fetching text from the server."
        );
        setButtonText("Start Recording");
        setIsRecording(false);
      }
    } else {
      // Cancel ongoing recording
      setButtonText("Start Recording");
      setReferenceText("Recording stopped...!!");
      setIsRecording(false);
    }
  };
  useEffect(() => {
        const socket = io(API);
    socket.on("error_update", (data) => {
      toast.error(data.message);
      // STOP everything immediately
      setIsRecording(false);
      setButtonText("Start Recording");
      setReferenceText("");
    });
    return () => socket.disconnect();

  }, []);

  /**
   * .adc-root is pinned between the navbar and the footer (see the .adc-root
   * block in audioDataCollection.css for why it has to be `position: fixed`).
   * Both bars change height with the window - the navbar stacks its rows below
   * 768px, the footer shrinks its type below 970px - so their heights are
   * measured from the live elements instead of hard-coded, and the page fits
   * every screen size and browser zoom level.
   */
  useEffect(() => {
    const nav = document.querySelector(".navbar");
    const footer = document.querySelector(".footer");

    const sync = () => {
      const el = rootRef.current;
      if (!el) return;

      if (nav) {
        const h = Math.ceil(nav.getBoundingClientRect().height);
        if (h > 0) el.style.setProperty("--adc-nav-h", `${h}px`);
      }

      if (footer) {
        // The footer's own box is short; .footer-color-boxes is absolutely
        // positioned and pokes out above it, so the height the page has to
        // clear is measured from the topmost of the two, not from the footer.
        let top = footer.getBoundingClientRect().top;
        const boxes = footer.querySelector(".footer-color-boxes");
        if (boxes) top = Math.min(top, boxes.getBoundingClientRect().top);
        const h = Math.ceil(window.innerHeight - top);
        if (h > 0) el.style.setProperty("--adc-footer-h", `${h}px`);
      }
    };

    sync();

    // ResizeObserver catches the bars re-flowing (navbar stacking, status text
    // changing length); resize covers zoom and window changes that leave the
    // bars' own boxes untouched.
    const ro =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(sync) : null;
    if (ro) {
      if (nav) ro.observe(nav);
      if (footer) ro.observe(footer);
    }
    window.addEventListener("resize", sync);

    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener("resize", sync);
    };
  }, []);

  return (
    <>
    {/* outside .adc-root: the root is a fixed box, and a fixed toast container
        inside it would be positioned against the root instead of the viewport */}
    <ToastWrapper />

    <div className="adc-root" ref={rootRef}>
      <div className="adc-shell">
        {/* Navigation buttons for different data collection modes */}
        <aside className="adc-rail">
          <span className="adc-rail-mark"></span>

          <div className="adc-nav-item is-current">
            <div className="nav-tooltip-wrapper adc-tile is-current">
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
            <span className="adc-nav-label">Audio</span>
          </div>

          <div className="adc-nav-item">
            <div className="nav-tooltip-wrapper adc-tile">
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
            <span className="adc-nav-label">Vision</span>
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
{/* 
          <div className="adc-nav-item is-locked">
            <div
              className="nav-tooltip-wrapper adc-tile is-locked disabled-tab"
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
                Sensor<br />(Under <br />Development)
              </div>
            </div>
            <span className="adc-nav-label">Sensor</span>
          </div> */}
        </aside>

        {/* Visual feedback area for recording */}
        <section className="adc-stage">
          <div className="adc-head">
            <div>
              <p className="adc-eyebrow">Data Collection</p>
              <h2 className="adc-title">Audio Capture Studio</h2>
              <p className="adc-sub">
                Live monitor for the microphone capture session.
              </p>
            </div>
            <span className={`adc-chip${isRecording ? " is-live" : ""}`}>
              <span className="adc-dot"></span>
              {isRecording ? "Session Active" : "Standby"}
            </span>
          </div>

          <div className="adc-dial-zone">
            <div
              className={`${referenceText === "Recording"
                ? "circle-animation circle"
                : "circle"
                }`}
            >
              {!referenceText && (
                <img
                  src="assets/Ambient_wave.png"
                  alt="Breathing Effect"
                  className="breathing-image"
                  draggable="false"
                  onContextMenu={(e) => e.preventDefault()}
                />
              )}
              <div
                ref={progressCricle}
                className="progress-circle-static"
              ></div>
              <div className="text-container">
                <p className="reference-text">
                  {referenceText}
                  {referenceText === "Recording" && <span>{animatedDots}</span>}
                </p>
                {referenceText === "Recording" && (
                  <p className="reference-text">{countdownText}</p>
                )}
              </div>
            </div>
          </div>

          <div className="adc-meta">
            <div className="adc-meta-item">
              <p className="adc-meta-k">Input Device</p>
              <p className="adc-meta-v">{inputValues.select_device || "—"}</p>
            </div>
            <div className="adc-meta-item">
              <p className="adc-meta-k">Sample Rate</p>
              <p className="adc-meta-v">{inputValues.frequency} kHz</p>
            </div>
            <div className="adc-meta-item">
              <p className="adc-meta-k">Samples</p>
              <p className="adc-meta-v">{inputValues.number_of_samples || "—"}</p>
            </div>
            <div className="adc-meta-item">
              <p className="adc-meta-k">Duration</p>
              <p className="adc-meta-v">
                {inputValues.duration_of_sample
                  ? `${inputValues.duration_of_sample} s`
                  : "—"}
              </p>
            </div>
          </div>
        </section>

        {/* Form section for input parameters */}
        <section className="adc-panel">
          <div className="adc-panel-head">
            <p className="adc-eyebrow">Session Setup</p>
            <h2 className="adc-title">Collect Voice Sample from Microphone</h2>
            <p className="adc-sub">
              Define the speaker, target word and capture parameters, then start
              the recording.
            </p>
          </div>

          <form className="adc-form">
            <div className="adc-field is-full">
              <label className="adc-label" htmlFor="person_name">
                Person Name
              </label>
              <input
                type="text"
                spellCheck="false"
                id="person_name"
                className="adc-input"
                placeholder="e.g. Alex"
                value={inputValues.person_name}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "word_name")}
                disabled={isRecording}
              />
              <div className="adc-tip">Enter the name of the person whose data is being recorded</div>
            </div>

            <div className="adc-field is-full">
              <label className="adc-label" htmlFor="word_name">
                Word to Record
              </label>
              <input
                type="text"
                spellCheck="false"
                id="word_name"
                className="adc-input"
                placeholder="e.g. Hello"
                value={inputValues.word_name}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "duration_of_sample")}
                disabled={isRecording}
              />
              <div className="adc-tip">Enter the word or phrase to be recorded</div>
            </div>

            <div className="adc-field">
              <label className="adc-label" htmlFor="duration_of_sample">
                Duration (seconds)
              </label>
              <input
                type="text"
                inputMode="numeric"
                id="duration_of_sample"
                className="adc-input"
                placeholder="e.g. 2"
                minLength={1}
                onWheel={(event) => event.currentTarget.blur()}
                value={inputValues.duration_of_sample}
                onKeyDown={(e) => handleKeyDown(e, "number_of_samples")}
                onChange={handleInputChange}
                disabled={isRecording}
              />
              <div className="adc-tip">Duration in seconds for each recording sample</div>
            </div>

            <div className="adc-field is-right">
              <label className="adc-label" htmlFor="number_of_samples">
                Number of Samples
              </label>
              <input
                type="number"
                id="number_of_samples"
                className="adc-input"
                placeholder="e.g. 10"
                onWheel={(event) => event.currentTarget.blur()}
                value={inputValues.number_of_samples}
                onChange={handleInputChange}
                onKeyDown={(e) => handleKeyDown(e, "frequency")}
                disabled={isRecording}
              />
              <div className="adc-tip">Specify how many recording samples to collect</div>
            </div>

            <div className="adc-field">
              <span className="adc-label">Input Device</span>
              {/* id + tabIndex keep the Enter-to-next-field chain landing here */}
              <div id="select_device" tabIndex={-1} className="adc-dd-slot">
                <CustomDropdown
                  value={inputValues.select_device || "GPX-I2S"}
                  onChange={(val) =>
                    setInputValues((prevValues) => ({
                      ...prevValues,
                      select_device: val,
                    }))
                  }
                  options={[
                    { value: "Laptop", label: "Laptop MIC (Unavailable)", disabled: true },
                    { value: "GPX-I2S", label: "GPX-I2S MIC" },
                  ]}
                />
              </div>
              <div className="adc-tip">Select the input device for recording</div>
            </div>

            <div className="adc-field is-right">
              <span className="adc-label">Sampling Frequency</span>
              <div id="frequency" tabIndex={-1} className="adc-dd-slot">
                <CustomDropdown
                  value={inputValues.frequency || 8}
                  onChange={(val) =>
                    setInputValues((prevValues) => ({
                      ...prevValues,
                      frequency: val,
                    }))
                  }
                  options={[
                    { value: 1, label: "1 kHz (Disabled)", disabled: true },
                    { value: 2, label: "2 kHz (Disabled)", disabled: true },
                    { value: 4, label: "4 kHz (Disabled)", disabled: true },
                    { value: 8, label: "8 kHz" },
                    { value: 16, label: "16 kHz (Disabled)", disabled: true },
                  ]}
                />
              </div>
              <div className="adc-tip">Sampling frequency of the audio signal</div>
            </div>
          </form>

          <div className="adc-actions">
            <Button
              text={buttonText}
              className="recording-button"
              onClick={handleButtonClick}
              disabled={isRecording}
            />
          </div>
        </section>
      </div>
    </div>
    </>
  );
};

export default AudioDataCollection;


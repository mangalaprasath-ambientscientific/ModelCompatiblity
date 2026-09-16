import React from "react";
import { HashRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import Navbar from "./components/navbar";
import Footer from "./components/Footer";
import AgentGate from "./components/AgentGate";
import Preprocessing from "./pages/Preprocessing";
import ModelCompatibility from "./pages/ModelCompatibility";
import ModelEmbeddedIntegration from "./pages/ModelEmbeddedIntegration";
import ModelDeployment from "./pages/ModelDeployment";
import RunningDebugging from "./pages/DebuggerPage";
import HowItWorks from "./pages/howItworks";
import AudioDataCollection from "./pages/AudioDataCollection";
import ResultLog from "./pages/resultLog";   
import Body from "./components/body";
import SensorDataCollection from "./pages/sensorDataCollection";
import VideoDataCollection from "./pages/VideoDataCollection";
import "./App.css"; 
import Flasher from "./pages/Flasher";
  
function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        {/* Pages that only compute or only read files are mounted directly:
            they have to work on a machine with no agent installed, which is
            the point of hosting the UI. Anything that reaches the J-Link over
            127.0.0.1 goes inside <AgentGate>, which renders the page only
            once the local agent answers and is able to flash - and otherwise
            explains which of "not installed", "browser blocked it", "too old"
            or "toolchain incomplete" actually happened. The `feature` string
            is quoted back to the user in that explanation. */}
        <Route path="/LandingPage" element={<Body />} />
        <Route path="/howitworks" element={<HowItWorks />} />
        <Route
          path="/audiodatacollection"
          element={
            <AgentGate feature="Audio data collection">
              <AudioDataCollection />
            </AgentGate>
          }
        />
        <Route
          path="/videodatacollection"
          element={
            <AgentGate feature="Video data collection">
              <VideoDataCollection />
            </AgentGate>
          }
        />
        <Route
          path="/sensordatacollection"
          element={
            <AgentGate feature="Sensor data collection">
              <SensorDataCollection />
            </AgentGate>
          }
        />
        <Route path="/preprocessing" element={<Preprocessing />} />
        <Route path="/modelcompatibility" element={<ModelCompatibility />} />
        <Route path="/modelembbededintegration" element={<ModelEmbeddedIntegration />} />
        <Route
          path="/runningdebugging"
          element={
            <AgentGate feature="Running and debugging">
              <RunningDebugging />
            </AgentGate>
          }
        />
        <Route path="/modeldeployment" element={<ModelDeployment />} />
        <Route
          path="/flasher"
          element={
            <AgentGate feature="Flashing">
              <Flasher />
            </AgentGate>
          }
        />
        <Route path="/resultlog" element={<ResultLog />} />
        <Route path="*" element={<Navigate to="/LandingPage" />} /> 
      </Routes>
      <Footer />
    </Router> 
  );
}
 
export default App;
 
 
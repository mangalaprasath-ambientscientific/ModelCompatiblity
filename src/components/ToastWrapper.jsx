import React from "react";
import { Slide, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const ToastWrapper = () => {
  return (
    <div id="toast-container-wrapper">
      <ToastContainer
        theme="dark"
        position="top-center"
        autoClose={7000}
        hideProgressBar={false}
        closeOnClick={false}
        draggable={false}
        transition={Slide}
        pauseOnHover
        toastClassName="custom-toast" 
        containerClassName="" 
        style={{ marginTop: "30px" }}
        closeButton={false}
      /> 
    </div>
  );
};

export default ToastWrapper;

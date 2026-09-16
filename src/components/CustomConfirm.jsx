import { confirmAlert } from "react-confirm-alert";
import "react-confirm-alert/src/react-confirm-alert.css";

export const customPrompt = ({ title, message, defaultValue, onConfirm }) => {
  let inputValue = defaultValue || "";

  confirmAlert({
    customUI: ({ onClose }) => (
      <div
        style={{
          backgroundColor: "#151515",
          color: "#ECECEC",
          padding: "20px",
          borderRadius: "8px",
          minWidth: "400px", // ðŸ‘ˆ Adjust this value as needed
          maxWidth: "600px", // Optional, to prevent it from getting too wide
          fontFamily: "var(--es-sans)",
        }}
      >
        <h1>{title}</h1>
        <p>{message}</p>
        <input
          type="text"
          defaultValue={defaultValue}
          autoFocus
          onChange={(e) => (inputValue = e.target.value)}
          style={{
            marginTop: "10px",
            width: "95%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #333",
            backgroundColor: "#1e1e1e",
            color: "#ECECEC",
            fontFamily: "var(--es-sans)",
          }}
        />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "20px",
          }}
        >
          <button
            onClick={() => {
              onConfirm(inputValue);
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
            OK
          </button>
          <button
            onClick={onClose}
            style={{
              backgroundColor: "red",
              color: "#ECECEC",
              fontFamily: "var(--es-sans)",
              padding: "10px 20px",
              borderRadius: "4px",
              border: "none",
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    ),
  });
};

export const customConfirm = ({ title, message, itemName, onConfirm }) => {
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
        <h1>{title}</h1>
        <p>
          {message}
          <strong style={{ display: "block", marginTop: "10px" }}>
            {itemName}
          </strong>
        </p>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "20px",
          }}
        >
          <button
            onClick={() => {
              onConfirm();
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
            onClick={onClose}
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

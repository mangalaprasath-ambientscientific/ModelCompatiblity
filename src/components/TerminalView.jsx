import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const socket = io("http://127.0.0.1:9262");

const TerminalView = ({ selectedPath = "", isVisible, isRunning, setIsRunning }) => {
  const terminalRef = useRef(null);
  const inputRef = useRef(null);
  const [input, setInput] = useState("");
  const [cwd, setCwd] = useState(selectedPath || "C:/");
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionIndex, setSuggestionIndex] = useState(0);
  const [lastCommandType, setLastCommandType] = useState("");
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);


  const appendOutput = (text) => {
    const div = document.createElement("div");
    div.textContent = text;
    terminalRef.current.appendChild(div);
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
  };

  const handleSubmit = () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isRunning) return;

    setHistory((prev) => [...prev, trimmedInput]);
    setHistoryIndex(-1);

    appendOutput(`${cwd}> ${trimmedInput}`);
    let normalizedInput = trimmedInput;
    if (trimmedInput.toLowerCase() === "cd..") {
      normalizedInput = "cd ..";
    }
    const [command, ...args] = normalizedInput.split(" ");
    const argText = args.join(" ");

    switch (command.toLowerCase()) {
      case "cls":
      case "clear":
        terminalRef.current.innerHTML = "";
        break;
      case "help":
        appendOutput("Available commands:");
        appendOutput("  cls / clear    - Clear terminal");
        appendOutput("  help           - Show help");
        appendOutput("  cwd            - Show current directory");
        appendOutput("  cd [path]      - Change directory");
        appendOutput("  mkdir [folder] - Create new folder");
        appendOutput("  ls/dir             - List contents");
        appendOutput("  clearinput     - Clear input box only");
        break;
      case "cwd":
        appendOutput(cwd);
        break;
      case "clearinput":
        setInput("");
        break;
      case "cd":
        if (!argText) {
          appendOutput("Usage: cd [path]");
        } else {
          socket.emit("cd", { path: argText, cwd });
          setIsRunning(true);
        }
        break;
      case "mkdir":
        if (!argText) {
          appendOutput("Usage: mkdir [folder_name]");
        } else {
          socket.emit("mkdir", { path: cwd, folder_name: argText });
          setIsRunning(true);
        }
        break;
      case "dir":
      case "ls":
        socket.emit("ls", { path: cwd });
        setIsRunning(true);
        break;
      default:
        socket.emit("run_command", {
          command: trimmedInput,
          cwd: cwd
        });
        setIsRunning(true);
        break;
    }

    setInput("");
    setSuggestions([]);
    setSuggestionIndex(0);
    setLastCommandType("");
    focusInput();
  };

  const handleTab = () => {
    const tokens = input.trim().split(" ");
    const command = tokens[0].toLowerCase();
    const partial = tokens.slice(1).join(" ");

    if (command === "cd") {
      socket.emit("autocomplete_folder", { cwd, partial });
      setLastCommandType("cd");
    } else if (command === "python") {
      socket.emit("autocomplete_python", { cwd, partial });
      setLastCommandType("python");
    }
  };

  const cycleSuggestion = () => {
    if (suggestions.length === 0) return;

    let base = input;
    if (lastCommandType === "cd") base = "cd ";
    else if (lastCommandType === "python") base = "python ";

    const nextSuggestion = suggestions[suggestionIndex];
    setInput(`${base}${nextSuggestion}`);
    setSuggestionIndex((prev) => (prev + 1) % suggestions.length);
  };

  const focusInput = () => {
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  useEffect(() => {
    if (selectedPath) {
      const dirPath = selectedPath.replace(/\\[^\\]+$/, ''); // Remove filename if present
      setCwd(dirPath);
      // Notify backend of initial directory
      socket.emit('set_initial_directory', { path: dirPath });
    }
  }, [selectedPath]);

  useEffect(() => {
    socket.on("terminal_output", (data) => {
      if (data.output) appendOutput(data.output.trim());
      if (data.cwd) setCwd(data.cwd);
      focusInput();
    });

    socket.on("script_completed", () => {
      setIsRunning(false);
      focusInput();
    });

    socket.on("autocomplete_suggestions", (data) => {
      if (Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        setSuggestions(data.suggestions);
        setSuggestionIndex(0);
        cycleSuggestion();
      }
    });


    return () => {
      socket.off("terminal_output");
      socket.off("autocomplete_suggestions");
      socket.off("script_completed");
      socket.off("connect");
    };
  }, [suggestions, suggestionIndex]);

  return (
    <div className="terminal-output" onClick={focusInput}>
      <div ref={terminalRef} className="terminal-view-content" />
      <div className="terminal-input-wrapper">
        <span className="terminal-cwd">{isRunning ? "" : `${cwd}> `}</span>
        {!isRunning && (
          <input
            ref={inputRef}
            value={input}
            disabled={isRunning}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSubmit();
              } else if (e.key === "Tab") {
                e.preventDefault();
                if (suggestions.length > 0) {
                  cycleSuggestion();
                } else {
                  handleTab();
                }
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                if (history.length > 0 && historyIndex < history.length - 1) {
                  const newIndex = historyIndex + 1;
                  setHistoryIndex(newIndex);
                  setInput(history[history.length - 1 - newIndex]);
                }
              } else if (e.key === "ArrowDown") {
                e.preventDefault();
                if (historyIndex > 0) {
                  const newIndex = historyIndex - 1;
                  setHistoryIndex(newIndex);
                  setInput(history[history.length - 1 - newIndex]);
                } else {
                  setHistoryIndex(-1);
                  setInput("");
                }
              }
            }}

            className="terminal-input"
          />
        )}
      </div>
    </div>

  );
};
export default TerminalView;
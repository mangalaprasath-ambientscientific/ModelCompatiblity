import React, { useState, useRef, useEffect } from "react";

/**
 * CustomDropdown — themed, animated replacement for <select>.
 *
 * Props:
 *   value        current value (string)
 *   onChange     (val) => void
 *   options      [{ value, label, disabled? }]  (a disabled option is shown
 *                but cannot be picked, like <option disabled>)
 *   placeholder  shown when nothing selected
 *   disabled     bool
 */
export default function CustomDropdown({
  value,
  onChange,
  options = [],
  placeholder = "Select...",
  disabled = false,
}) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const rootRef = useRef(null);

  const selected = options.find((o) => o.value === value);

  // close on outside click
  useEffect(() => {
    const onDocClick = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  // keyboard: escape / arrows / enter
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === "Escape") {
        setOpen(false);
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((i) => Math.min(i + 1, options.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((i) => Math.max(i - 1, 0));
      } else if (e.key === "Enter" && activeIndex >= 0) {
        e.preventDefault();
        if (options[activeIndex].disabled) return;
        onChange(options[activeIndex].value);
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, activeIndex, options, onChange]);

  const toggle = () => {
    if (disabled) return;
    setOpen((o) => !o);
    setActiveIndex(options.findIndex((o) => o.value === value));
  };

  const pick = (val) => {
    onChange(val);
    setOpen(false);
  };

  return (
    <div
      className={`dd ${open ? "open" : ""} ${disabled ? "dd-disabled" : ""}`}
      ref={rootRef}
    >
      <button
        type="button"
        className={`dd-trigger ${selected ? "" : "dd-placeholder"}`}
        onClick={toggle}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled}
      >
        <span className="dd-value">{selected ? selected.label : placeholder}</span>
        <svg
          className="dd-arrow"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      <div className="dd-panel" role="listbox">
        {options.map((opt, i) => (
          <button
            type="button"
            key={opt.value}
            role="option"
            aria-selected={opt.value === value}
            className={`dd-option ${opt.value === value ? "selected" : ""} ${
              i === activeIndex ? "active" : ""
            } ${opt.disabled ? "is-disabled" : ""}`}
            style={{ "--i": i }}
            disabled={opt.disabled}
            onClick={() => { if (!opt.disabled) pick(opt.value); }}
            onMouseEnter={() => setActiveIndex(i)}
          >
         <span className="dd-option-label">{opt.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// components/DriverPicker.tsx
//
// What it does: a search box for choosing a driver. Type part of a name and
//   pick from the matches (mouse, touch, or arrow keys and Enter).
// Which files use it: the race, head to head and drivers pages.

"use client";

import { useState } from "react";

import type { Driver } from "@/lib/data";

type Props = {
  drivers: Driver[]; // the drivers that may be chosen
  onPick: (driver: Driver) => void;
  placeholder?: string;
};

/** Lower case with accents removed, so "hakkinen" finds "Häkkinen". */
function plain(text: string): string {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

export default function DriverPicker({ drivers, onPick, placeholder = "Search for a driver..." }: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);

  // With nothing typed, offer the best known names first (most race starts).
  const matches = (query
    ? drivers.filter((driver) => plain(driver.name).includes(plain(query)))
    : [...drivers].sort((a, b) => b.starts - a.starts)
  ).slice(0, 30);

  function pick(driver: Driver) {
    onPick(driver);
    setQuery("");
    setOpen(false);
  }

  function onKey(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown") setCursor((c) => Math.min(c + 1, matches.length - 1));
    else if (event.key === "ArrowUp") setCursor((c) => Math.max(c - 1, 0));
    else if (event.key === "Enter" && matches[cursor]) pick(matches[cursor]);
    else if (event.key === "Escape") setOpen(false);
    else return;
    event.preventDefault();
  }

  return (
    <div className="picker">
      <input
        value={query}
        placeholder={placeholder}
        aria-label={placeholder}
        onChange={(event) => {
          setQuery(event.target.value);
          setCursor(0);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        // The short delay lets a click on the list register before it closes.
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={onKey}
      />
      {open && matches.length > 0 && (
        <div className="picker-list">
          {matches.map((driver, position) => (
            <button
              key={driver.id}
              className={position === cursor ? "cursor" : ""}
              onMouseDown={(event) => event.preventDefault()} // keep the box focused
              onClick={() => pick(driver)}
            >
              <span>{driver.name}</span>
              <span className="muted small num">
                {driver.years[0]}–{driver.years[1]}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

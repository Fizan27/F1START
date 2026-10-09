// lib/data.ts
//
// What it does: describes the shape of every JSON file the website reads, and
//   loads them. All of them are written by export_web.py, validate_model.py
//   and download_circuits.py (the Python side); nothing here is typed by hand.
// What it reads: the files in public/data/.
// Which files use it: every page.

"use client";

import { useEffect, useState } from "react";

export type Driver = {
  id: string;
  name: string;
  rank: number; // 1 = highest peak skill of all drivers
  rankLow: number; // the rank could plausibly be anywhere from here...
  rankHigh: number; // ...to here (middle 90%)
  skill: number; // peak skill: percent of lap time quicker than an average newcomer of today
  sd: number; // uncertainty of the skill (one standard deviation)
  peak: [number, number]; // first and last season of the peak
  years: [number, number]; // first and last season raced
  starts: number;
  wins: number;
  mates: number | null; // how many different teammates
  consistency: number; // 0 to 100, 50 = typical
};

// [teammate id, from year, to year, races won, races compared, qualifyings won, qualifyings compared]
export type MateRecord = [string, number, number, number, number, number, number];

export type DriverDetail = {
  curve: [number, number, number][]; // [season, skill, uncertainty]
  mates: MateRecord[];
  teams: string[];
  crashRatio: number | null;
  lapSpreadRatio: number | null;
};

export type HeadToHeads = { drivers: string[]; best: number[]; low: number[]; high: number[] };

// [driver id, equal car points, equal car title chance %, real points (today's system)]
export type SeasonRow = [string, number, number, number];
export type Season = { year: number; races: number; table: SeasonRow[] };

export type Circuit = {
  id: string;
  name: string;
  country: string;
  year: number;
  raceLaps: number;
  lapSeconds: number;
  points?: [number, number][];
};

export type SimSettings = {
  formSd: number;
  qualiNoise: number;
  lapNoise: number;
  mistakeChance: number;
  mistakeSeconds: number;
  crashChance: number;
  gridGapSeconds: number;
  followGapSeconds: number;
  passMarginEasy: number;
  passMarginHard: number;
};

export type Settings = { sim: SimSettings; raceSlope: number; difficulty: Record<string, number> };

// Each file is downloaded once and then shared by every page that asks for it.
const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(`/data/${path}`).then((response) => response.json()));
  }
  return cache.get(path) as Promise<T>;
}

/** Load one JSON file from public/data. Returns null until it has arrived. */
export function useData<T>(path: string | null): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    let current = true; // ignore an answer that arrives after the page moved on
    setData(null);
    if (path) load<T>(path).then((value) => current && setData(value));
    return () => {
      current = false;
    };
  }, [path]);
  return data;
}

/** The 90% range of a rating: 1.645 standard deviations either side. */
export function skillRange(driver: { skill: number; sd: number }): [number, number] {
  return [driver.skill - 1.645 * driver.sd, driver.skill + 1.645 * driver.sd];
}

/** "+0.92" or "-0.10": skill always shown with its sign. */
export function signed(value: number, digits = 2): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

/** Where a pair sits in the head to head lists, which hold each pair once (row < column). */
export function pairPosition(row: number, column: number, size: number): number {
  return row * size - (row * (row + 1)) / 2 + (column - row - 1);
}

/** Chance (whole percent) that A beats B, with its range. Null if either is not in the list. */
export function headToHead(h2h: HeadToHeads, a: string, b: string): { best: number; low: number; high: number } | null {
  const i = h2h.drivers.indexOf(a);
  const j = h2h.drivers.indexOf(b);
  if (i < 0 || j < 0 || i === j) return null;
  if (i < j) {
    const k = pairPosition(i, j, h2h.drivers.length);
    return { best: h2h.best[k], low: h2h.low[k], high: h2h.high[k] };
  }
  // Stored the other way round: A's chance is 100 minus B's, and the range flips.
  const k = pairPosition(j, i, h2h.drivers.length);
  return { best: 100 - h2h.best[k], low: 100 - h2h.high[k], high: 100 - h2h.low[k] };
}

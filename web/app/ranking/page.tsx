// app/ranking/page.tsx
//
// What it does: the all time ranking. Every driver with their peak skill and
//   its uncertainty range drawn as a bar, filterable by era and by a minimum
//   number of races.
// What it reads: drivers.json from public/data.
// Which files use it: none (it is a page).

"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { type Driver, signed, skillRange, useData } from "@/lib/data";

const ERAS = ["All", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"];
const START_CHOICES = [1, 10, 30, 50, 100, 200];

/** The decade a driver's peak falls in, for example "1990s". */
function peakEra(driver: Driver): string {
  return `${Math.floor(driver.peak[0] / 10) * 10}s`;
}

export default function RankingPage() {
  const drivers = useData<Driver[]>("drivers.json");
  const router = useRouter();
  const [era, setEra] = useState("All");
  const [minStarts, setMinStarts] = useState(50);

  if (!drivers) return <p className="muted">Loading...</p>;

  const shown = drivers.filter((d) => d.starts >= minStarts && (era === "All" || peakEra(d) === era));
  // One shared scale for every row, so bars can be compared down the page.
  const ranges = shown.map(skillRange);
  const scaleLow = Math.min(-0.2, ...ranges.map(([low]) => low));
  const scaleHigh = Math.max(0.5, ...ranges.map(([, high]) => high));

  return (
    <>
      <h1>All time ranking</h1>
      <p className="lead">
        Peak skill: how much quicker each driver was, at their best three seasons in a row, than an
        average newcomer of today in the same car. The bar is the range the true value probably lies
        in. Where two bars overlap, the model cannot honestly separate the two drivers.
      </p>

      <div className="row" style={{ marginBottom: 16 }}>
        {ERAS.map((choice) => (
          <button key={choice} className={`chip ${era === choice ? "on" : ""}`} onClick={() => setEra(choice)}>
            {choice}
          </button>
        ))}
        <label className="row small muted" style={{ marginLeft: "auto" }}>
          At least
          <select value={minStarts} onChange={(event) => setMinStarts(Number(event.target.value))}>
            {START_CHOICES.map((choice) => (
              <option key={choice} value={choice}>
                {choice} {choice === 1 ? "race" : "races"}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="card table-scroll" style={{ padding: 6 }}>
        <table>
          <thead>
            <tr>
              <th className="right">#</th>
              <th>Driver</th>
              <th className="hide-phone">Peak</th>
              <th style={{ width: "34%" }}>
                Skill and 90% range <span className="num">({signed(scaleLow, 1)}% to {signed(scaleHigh, 1)}%)</span>
              </th>
              <th className="right">Skill</th>
              <th className="right hide-phone">Could rank</th>
              <th className="right hide-phone">Races</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((driver, place) => (
              <tr key={driver.id} className="clickable" onClick={() => router.push(`/drivers/?id=${driver.id}`)}>
                <td className="right num muted">{place + 1}</td>
                <td style={{ fontWeight: 600 }}>{driver.name}</td>
                <td className="num muted hide-phone">
                  {driver.peak[0]}–{driver.peak[1]}
                </td>
                <td>
                  <IntervalBar driver={driver} low={scaleLow} high={scaleHigh} />
                </td>
                <td className="right num">{signed(driver.skill)}%</td>
                <td className="right num muted hide-phone">
                  {driver.rankLow}–{driver.rankHigh}
                </td>
                <td className="right num muted hide-phone">{driver.starts}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {shown.length === 0 && <p className="muted" style={{ padding: 12 }}>No drivers match these filters.</p>}
      </div>
      <p className="muted small" style={{ marginTop: 10 }}>
        {shown.length} drivers shown. "Could rank" is among all {drivers.length} rated drivers, whatever
        the filters. Tap a driver for their card.
      </p>
    </>
  );
}

/** A dot at the best estimate on a line spanning the 90% range, with zero marked. */
function IntervalBar({ driver, low, high }: { driver: Driver; low: number; high: number }) {
  const [from, to] = skillRange(driver);
  const x = (value: number) => ((value - low) / (high - low)) * 100;
  const label = `${driver.name}: ${signed(driver.skill)}%, range ${signed(from)}% to ${signed(to)}%`;
  return (
    <svg className="interval" viewBox="0 0 100 22" preserveAspectRatio="none" role="img" aria-label={label}>
      <title>{label}</title>
      <line x1={x(0)} x2={x(0)} y1={2} y2={20} stroke="var(--line-strong)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      <line x1={x(from)} x2={x(to)} y1={11} y2={11} stroke="var(--series-1)" strokeOpacity={0.45} strokeWidth={6} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      <line x1={x(driver.skill)} x2={x(driver.skill)} y1={11} y2={11} stroke="var(--ink)" strokeWidth={9} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// app/seasons/page.tsx
//
// What it does: the equal car championships. For any season since 1950, shows
//   who would have been expected to win the title if every driver that year
//   had driven the same car, next to what really happened.
// What it reads: championships.json and drivers.json from public/data.
// Which files use it: none (it is a page).

"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { type Driver, type Season, useData } from "@/lib/data";

export default function SeasonsPage() {
  const seasons = useData<Season[]>("championships.json");
  const drivers = useData<Driver[]>("drivers.json");
  const router = useRouter();
  const [year, setYear] = useState(2021);

  if (!seasons || !drivers) return <p className="muted">Loading...</p>;
  const season = seasons.find((s) => s.year === year) ?? seasons[seasons.length - 1];
  const nameOf = (id: string) => drivers.find((d) => d.id === id)?.name ?? id;

  // Real finishing order among the drivers shown, by today's points system.
  const realOrder = [...season.table].sort((a, b) => b[3] - a[3]).map((row) => row[0]);
  const mostPoints = Math.max(...season.table.map((row) => Math.max(row[1], row[3])));

  return (
    <>
      <h1>Equal car championships</h1>
      <p className="lead">
        Every season replayed 1,000 times with its real calendar and its real drivers, at the level each
        was driving that year, but with everyone in the same car. Both columns use today's points
        (25 for a win), so they compare like with like.
      </p>

      <div className="row" style={{ marginBottom: 16 }}>
        <button onClick={() => setYear(Math.max(seasons[0].year, year - 1))} aria-label="Previous season">←</button>
        <select value={season.year} onChange={(event) => setYear(Number(event.target.value))} aria-label="Season">
          {[...seasons].reverse().map((s) => (
            <option key={s.year} value={s.year}>{s.year}</option>
          ))}
        </select>
        <button onClick={() => setYear(Math.min(seasons[seasons.length - 1].year, year + 1))} aria-label="Next season">→</button>
        <span className="muted small">{season.races} races</span>
      </div>

      <div className="card table-scroll" style={{ padding: 6 }}>
        <table>
          <thead>
            <tr>
              <th className="right">#</th>
              <th>Driver</th>
              <th style={{ width: "36%" }}>Points per season: equal cars and real</th>
              <th className="right">Equal cars</th>
              <th className="right">Title chance</th>
              <th className="right">Real</th>
              <th className="right hide-phone">Real place</th>
            </tr>
          </thead>
          <tbody>
            {season.table.map(([id, points, title, real], place) => (
              <tr key={id} className="clickable" onClick={() => router.push(`/drivers/?id=${id}`)}>
                <td className="right num muted">{place + 1}</td>
                <td style={{ fontWeight: 600 }}>{nameOf(id)}</td>
                <td>
                  <PointsBars equal={points} real={real} most={mostPoints} name={nameOf(id)} />
                </td>
                <td className="right num">{points}</td>
                <td className="right num">{title.toFixed(title < 10 ? 1 : 0)}%</td>
                <td className="right num muted">{real}</td>
                <td className="right num muted hide-phone">{realOrder.indexOf(id) + 1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)" }} />Equal cars (average of 1,000 simulated seasons)</span>
        <span><i style={{ background: "var(--muted)" }} />What really happened</span>
      </div>
      <p className="muted small" style={{ marginTop: 10 }}>
        The top 12 by equal car points are shown; "real place" is among those 12. A driver far higher in
        equal cars than in reality was held back by their car that year, according to the model.
      </p>
    </>
  );
}

/** Two thin bars on one scale: equal car points above, real points below. */
function PointsBars({ equal, real, most, name }: { equal: number; real: number; most: number; name: string }) {
  const label = `${name}: ${equal} points in equal cars, ${real} in reality`;
  return (
    <div role="img" aria-label={label} title={label} style={{ display: "grid", gap: 2, minWidth: 140 }}>
      <div style={{ width: `${Math.max((equal / most) * 100, 0.5)}%`, height: 7, background: "var(--series-1)", borderRadius: "0 4px 4px 0" }} />
      <div style={{ width: `${Math.max((real / most) * 100, 0.5)}%`, height: 7, background: "var(--muted)", borderRadius: "0 4px 4px 0" }} />
    </div>
  );
}

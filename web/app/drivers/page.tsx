// app/drivers/page.tsx
//
// What it does: the driver card. For one driver: skill with its range,
//   consistency, peak years, the career curve (skill season by season) and
//   the record against each teammate.
//   The driver is chosen by the address: /drivers/?id=senna
// What it reads: drivers.json and details.json from public/data.
// Which files use it: none (it is a page). The ranking and head to head
//   pages link here.

"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import DriverPicker from "@/components/DriverPicker";
import { type Driver, type DriverDetail, signed, skillRange, useData } from "@/lib/data";

export default function DriversPage() {
  // Next.js needs a page that reads the address to be wrapped like this.
  return (
    <Suspense fallback={<p className="muted">Loading...</p>}>
      <DriverCard />
    </Suspense>
  );
}

function DriverCard() {
  const drivers = useData<Driver[]>("drivers.json");
  const details = useData<Record<string, DriverDetail>>("details.json");
  const router = useRouter();
  const id = useSearchParams().get("id") ?? "senna";

  if (!drivers || !details) return <p className="muted">Loading...</p>;
  const driver = drivers.find((d) => d.id === id) ?? drivers[0];
  const detail = details[driver.id];
  const [low, high] = skillRange(driver);
  const nameOf = (driverId: string) => drivers.find((d) => d.id === driverId)?.name ?? driverId;

  return (
    <>
      <div style={{ maxWidth: 420, marginBottom: 20 }}>
        <DriverPicker drivers={drivers} onPick={(d) => router.push(`/drivers/?id=${d.id}`)} placeholder="Find another driver..." />
      </div>

      <h1>{driver.name}</h1>
      <p className="lead">
        {driver.years[0]}–{driver.years[1]} · {driver.starts} starts · {driver.wins} {driver.wins === 1 ? "win" : "wins"} ·{" "}
        {detail.teams.join(", ")}
      </p>

      <div className="stats" style={{ marginBottom: 16 }}>
        <div className="stat">
          <b className="num">{signed(driver.skill)}%</b>
          <span>peak skill (range {signed(low)} to {signed(high)})</span>
        </div>
        <div className="stat">
          <b className="num">#{driver.rank}</b>
          <span>all time (could be {driver.rankLow} to {driver.rankHigh})</span>
        </div>
        <div className="stat">
          <b className="num">
            {driver.peak[0]}–{driver.peak[1]}
          </b>
          <span>peak years</span>
        </div>
        <div className="stat">
          <b className="num">{driver.consistency}</b>
          <span>consistency (50 is typical)</span>
        </div>
      </div>

      <div className="stack">
        <div className="card">
          <h2>Career curve</h2>
          <CareerChart curve={detail.curve} peak={driver.peak} />
          <p className="muted small" style={{ marginTop: 8 }}>
            Skill each season, in percent of lap time quicker than an average newcomer of today in the same
            car. The band is the 90% range. The brighter stretch marks the peak years.
          </p>
        </div>

        <div className="card">
          <h2>Teammate battles</h2>
          {detail.mates.length === 0 ? (
            <p className="muted">No teammates on record.</p>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Teammate</th>
                    <th className="hide-phone">Years</th>
                    <th>Races ahead</th>
                    <th>Qualifying ahead</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.mates.map(([mate, from, to, raceWins, races, qualiWins, qualis]) => (
                    <tr key={mate} className="clickable" onClick={() => router.push(`/drivers/?id=${mate}`)}>
                      <td style={{ fontWeight: 600 }}>{nameOf(mate)}</td>
                      <td className="num muted hide-phone">{from === to ? from : `${from}–${to}`}</td>
                      <td>
                        <Record won={raceWins} of={races} />
                      </td>
                      <td>
                        <Record won={qualiWins} of={qualis} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="muted small" style={{ marginTop: 8 }}>
            Races where either car broke down are not counted, so a breakdown never counts against a driver.
            {detail.crashRatio !== null && (
              <> Crashed out {detail.crashRatio.toFixed(2)}× as often as a typical driver of the same era.</>
            )}
          </p>
        </div>
      </div>
    </>
  );
}

/** "12 of 20" with a small bar showing the share. */
function Record({ won, of }: { won: number; of: number }) {
  if (of === 0) return <span className="muted">no data</span>;
  return (
    <div className="row" style={{ flexWrap: "nowrap" }}>
      <span className="num" style={{ minWidth: 62 }}>
        {won} of {of}
      </span>
      <div className="mini-bar" style={{ flex: 1 }} role="img" aria-label={`${won} of ${of}`}>
        <div style={{ width: `${(won / of) * 100}%` }} />
        <div style={{ width: `${(1 - won / of) * 100}%` }} />
      </div>
    </div>
  );
}

const WIDTH = 720;
const HEIGHT = 250;
const PAD = { left: 44, right: 14, top: 12, bottom: 26 };
const RANGE_WIDTH = 1.645; // standard deviations either side for a 90% range

/** The career curve: a line of skill by season with its 90% band. Hover for exact values. */
function CareerChart({ curve, peak }: { curve: [number, number, number][]; peak: [number, number] }) {
  const [hover, setHover] = useState<number | null>(null);

  const years = curve.map(([year]) => year);
  const [firstYear, lastYear] = [years[0], years[years.length - 1]];
  const lows = curve.map(([, skill, sd]) => skill - RANGE_WIDTH * sd);
  const highs = curve.map(([, skill, sd]) => skill + RANGE_WIDTH * sd);
  const bottom = Math.min(0, ...lows) - 0.05;
  const top = Math.max(0.2, ...highs) + 0.05;

  const x = (year: number) =>
    PAD.left + (lastYear === firstYear ? 0.5 : (year - firstYear) / (lastYear - firstYear)) * (WIDTH - PAD.left - PAD.right);
  const y = (value: number) => PAD.top + ((top - value) / (top - bottom)) * (HEIGHT - PAD.top - PAD.bottom);

  const line = curve.map(([year, skill]) => `${x(year)},${y(skill)}`).join(" ");
  const band = [
    ...curve.map(([year], i) => `${x(year)},${y(highs[i])}`),
    ...curve.map(([year], i) => `${x(year)},${y(lows[i])}`).reverse(),
  ].join(" ");
  const peakLine = curve.filter(([year]) => year >= peak[0] && year <= peak[1])
    .map(([year, skill]) => `${x(year)},${y(skill)}`).join(" ");

  // About five evenly spaced grid lines, and about six year labels.
  const step = top - bottom > 1.6 ? 0.5 : 0.25;
  const gridValues: number[] = [];
  for (let value = Math.ceil(bottom / step) * step; value <= top; value += step) gridValues.push(value);
  const yearStep = Math.max(1, Math.ceil(years.length / 6));

  function onMove(event: React.MouseEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const at = ((event.clientX - box.left) / box.width) * WIDTH;
    let nearest = 0;
    curve.forEach(([year], i) => {
      if (Math.abs(x(year) - at) < Math.abs(x(curve[nearest][0]) - at)) nearest = i;
    });
    setHover(nearest);
  }

  const shown = hover === null ? null : curve[hover];
  return (
    <div className="chart-wrap">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`} style={{ width: "100%", height: "auto", display: "block" }}
        role="img" aria-label="Skill by season" onMouseMove={onMove} onMouseLeave={() => setHover(null)}
      >
        {gridValues.map((value) => (
          <g key={value}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(value)} y2={y(value)}
              stroke={Math.abs(value) < 1e-9 ? "var(--line-strong)" : "var(--line)"} strokeWidth={1} />
            <text x={PAD.left - 8} y={y(value) + 4} textAnchor="end" className="axis-text">
              {signed(value, 2)}
            </text>
          </g>
        ))}
        {years.filter((_, i) => i % yearStep === 0).map((year) => (
          <text key={year} x={x(year)} y={HEIGHT - 6} textAnchor="middle" className="axis-text">
            {year}
          </text>
        ))}
        <polygon points={band} fill="var(--series-1)" fillOpacity={0.16} />
        <polyline points={line} fill="none" stroke="var(--series-1)" strokeOpacity={0.55} strokeWidth={2} strokeLinejoin="round" />
        <polyline points={peakLine} fill="none" stroke="var(--series-1)" strokeWidth={3.5} strokeLinejoin="round" strokeLinecap="round" />
        {curve.length === 1 && <circle cx={x(firstYear)} cy={y(curve[0][1])} r={5} fill="var(--series-1)" />}
        {shown && (
          <g>
            <line x1={x(shown[0])} x2={x(shown[0])} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="var(--muted)" strokeWidth={1} />
            <circle cx={x(shown[0])} cy={y(shown[1])} r={5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        )}
      </svg>
      {shown && (
        <div className="tooltip num" style={{ left: `${(x(shown[0]) / WIDTH) * 100}%`, top: `${(y(shown[1]) / HEIGHT) * 100}%` }}>
          <b>{shown[0]}</b>: {signed(shown[1])}%{" "}
          <span className="muted">
            ({signed(shown[1] - RANGE_WIDTH * shown[2])} to {signed(shown[1] + RANGE_WIDTH * shown[2])})
          </span>
        </div>
      )}
    </div>
  );
}

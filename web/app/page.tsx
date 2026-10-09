// app/page.tsx
//
// What it does: the fantasy race page. Pick drivers from any era and a
//   circuit, press start, and watch the race: cars move round the real track
//   outline while a timing tower shows the order and the gaps.
//   The race is simulated in the browser (lib/sim.ts) the moment you press
//   start, so every race is different. The animation then just plays back
//   the result, which is why "replay" shows exactly the same race again.
// What it reads: drivers.json, settings.json, circuits/index.json and the
//   chosen circuit's outline, all from public/data.
// Which files use it: none (it is the home page).

"use client";

import { useEffect, useRef, useState } from "react";

import DriverPicker from "@/components/DriverPicker";
import { type Circuit, type Driver, type Settings, useData } from "@/lib/data";
import { isOut, progressAt, type Race, raceLength, simulateRace, towerAt, type TowerRow } from "@/lib/sim";

const MAX_CARS = 20;
const REAL_SECONDS_PER_LAP = 3; // how long one lap takes to watch at 1x speed
const TOWER_ROW_HEIGHT = 32;
const TOWER_UPDATES_PER_SECOND = 6;

const LEGENDS = [
  "senna", "prost", "michael_schumacher", "hamilton", "max_verstappen", "fangio", "clark", "stewart",
  "lauda", "alonso", "vettel", "moss", "piquet", "mansell", "hakkinen", "raikkonen", "ascari",
  "jack_brabham", "emerson_fittipaldi", "gilles_villeneuve",
];

// Eight hues, used three ways (solid, white ring, hollow) to tell 20 cars
// apart. Every car also carries its number, so colour is never the only clue.
const HUES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

function carLook(car: number): { fill: string; stroke: string; strokeWidth: number } {
  const hue = HUES[car % HUES.length];
  const style = Math.floor(car / HUES.length);
  if (style === 0) return { fill: hue, stroke: "#0d0d0d", strokeWidth: 2 };
  if (style === 1) return { fill: hue, stroke: "#ffffff", strokeWidth: 4 };
  return { fill: "#1a1a19", stroke: hue, strokeWidth: 5 };
}

function surname(driver: Driver): string {
  return driver.name.split(" ").slice(1).join(" ") || driver.name;
}

function lapTimeText(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(3).padStart(6, "0")}`;
}

/** The x, y on the outline for a car that has done `progress` laps. */
function pointOnTrack(points: [number, number][], progress: number): [number, number] {
  const along = (((progress % 1) + 1) % 1) * points.length; // also handles cars waiting behind the line
  const from = Math.floor(along) % points.length;
  const to = (from + 1) % points.length;
  const part = along - Math.floor(along);
  return [
    points[from][0] + (points[to][0] - points[from][0]) * part,
    points[from][1] + (points[to][1] - points[from][1]) * part,
  ];
}

export default function RacePage() {
  const drivers = useData<Driver[]>("drivers.json");
  const settings = useData<Settings>("settings.json");
  const circuits = useData<Circuit[]>("circuits/index.json");

  const [entrants, setEntrants] = useState<Driver[]>([]);
  const [circuitId, setCircuitId] = useState("silverstone");
  const [distance, setDistance] = useState<"sprint" | "half" | "full">("sprint");
  const [phase, setPhase] = useState<"setup" | "racing" | "finished">("setup");
  const [race, setRace] = useState<Race | null>(null);
  const [speed, setSpeed] = useState(1);
  const [paused, setPaused] = useState(false);
  const [tower, setTower] = useState<TowerRow[]>([]);

  const outline = useData<Circuit>(`circuits/${circuitId}.json`);
  const circuit = circuits?.find((c) => c.id === circuitId);

  // These change 60 times a second, so they live outside React's state:
  // updating state that often would redraw the whole page every frame.
  const clock = useRef(0);
  const carElements = useRef<(SVGGElement | null)[]>([]);

  // Start with a grid of legends, once the driver list has arrived.
  useEffect(() => {
    if (drivers && entrants.length === 0) setEntrants(pickByIds(drivers, LEGENDS));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drivers]);

  // The animation loop: move the clock on, place the cars, refresh the tower.
  useEffect(() => {
    if (phase !== "racing" || !race || !outline?.points || !circuit) return;
    const points = outline.points;
    const end = raceLength(race);
    const raceSecondsPerRealSecond = circuit.lapSeconds / REAL_SECONDS_PER_LAP;
    let frame = 0;
    let previous = performance.now();
    let sinceTower = Infinity; // forces a tower update on the first frame

    function tick(now: number) {
      const elapsed = Math.min((now - previous) / 1000, 0.1); // a background tab must not jump ahead
      previous = now;
      if (!paused) clock.current += elapsed * raceSecondsPerRealSecond * speed;
      placeCars(race!, points, clock.current, carElements.current);
      sinceTower += elapsed;
      if (sinceTower > 1 / TOWER_UPDATES_PER_SECOND) {
        sinceTower = 0;
        setTower(towerAt(race!, clock.current));
      }
      if (clock.current > end + circuit!.lapSeconds * 0.3) {
        setTower(towerAt(race!, end));
        setPhase("finished");
        return;
      }
      frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [phase, race, outline, circuit, speed, paused]);

  if (!drivers || !settings || !circuits) return <p className="muted">Loading drivers...</p>;

  const laps = !circuit ? 15 : distance === "sprint" ? 15 : distance === "half" ? Math.round(circuit.raceLaps / 2) : circuit.raceLaps;

  function start() {
    if (!circuit || !settings) return;
    const difficulty = settings.difficulty[circuit.id] ?? 0.5;
    setRace(simulateRace(entrants, circuit.lapSeconds, laps, difficulty, settings.sim));
    replay();
  }

  function replay() {
    clock.current = 0;
    setPaused(false);
    setPhase("racing");
  }

  function skipToEnd() {
    if (race) clock.current = raceLength(race);
  }

  if (phase === "setup") {
    return (
      <Setup
        drivers={drivers} circuits={circuits} entrants={entrants} setEntrants={setEntrants}
        circuitId={circuitId} setCircuitId={setCircuitId} distance={distance} setDistance={setDistance}
        laps={laps} onStart={start}
      />
    );
  }

  const leaderLap = tower.length ? Math.min(tower[0].lapsDone + 1, race!.laps) : 1;
  return (
    <>
      <div className="race-bar">
        <div>
          <h3 style={{ margin: 0 }}>{circuit?.name}</h3>
          <div className="lap-counter num">
            {phase === "finished" ? "Chequered flag" : `Lap ${leaderLap} / ${race!.laps}`}
          </div>
        </div>
        <div className="row">
          {phase === "racing" && (
            <>
              <button onClick={() => setPaused(!paused)}>{paused ? "Resume" : "Pause"}</button>
              {[1, 2, 4].map((value) => (
                <button key={value} className={`chip ${speed === value ? "on" : ""}`} onClick={() => setSpeed(value)}>
                  {value}×
                </button>
              ))}
              <button onClick={skipToEnd}>Skip to end</button>
            </>
          )}
          {phase === "finished" && (
            <>
              <button onClick={replay}>Watch replay</button>
              <button className="primary" onClick={start}>Race again</button>
            </>
          )}
          <button onClick={() => setPhase("setup")}>Change grid</button>
        </div>
      </div>

      <div className="race-layout">
        <div className="stack">
          <div className="card track-card">
            {outline?.points ? (
              <Track points={outline.points} entrants={entrants} carElements={carElements} />
            ) : (
              <p className="muted">Loading track...</p>
            )}
          </div>
          {phase === "finished" && race && <Results race={race} entrants={entrants} />}
        </div>
        <div className="card">
          <h3>Timing</h3>
          <Tower rows={tower} entrants={entrants} race={race!} />
        </div>
      </div>
    </>
  );
}

function pickByIds(drivers: Driver[], ids: string[]): Driver[] {
  return ids.map((id) => drivers.find((driver) => driver.id === id)).filter((d): d is Driver => !!d);
}

/** Move every car's dot to where it is at this moment of the race. */
function placeCars(race: Race, points: [number, number][], time: number, elements: (SVGGElement | null)[]) {
  elements.forEach((element, car) => {
    if (!element) return;
    // Before the start, cars queue behind the line in grid order.
    const waiting = time <= race.lineTimes[car][0] ? -(race.grid[car] + 1) * 0.004 : 0;
    const [x, y] = pointOnTrack(points, progressAt(race, car, time) + waiting);
    element.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
    element.style.opacity = isOut(race, car, time) ? "0" : "1";
  });
}

// ---- The setup screen -------------------------------------------------------

type SetupProps = {
  drivers: Driver[];
  circuits: Circuit[];
  entrants: Driver[];
  setEntrants: (entrants: Driver[]) => void;
  circuitId: string;
  setCircuitId: (id: string) => void;
  distance: "sprint" | "half" | "full";
  setDistance: (distance: "sprint" | "half" | "full") => void;
  laps: number;
  onStart: () => void;
};

function Setup(props: SetupProps) {
  const { drivers, circuits, entrants, setEntrants } = props;
  const chosen = new Set(entrants.map((driver) => driver.id));
  const full = entrants.length >= MAX_CARS;

  const presets: [string, () => Driver[]][] = [
    ["Legends", () => pickByIds(drivers, LEGENDS)],
    ["All-time top 20", () => drivers.filter((d) => d.starts >= 50).slice(0, MAX_CARS)],
    ["Today's grid", () => {
      const latest = Math.max(...drivers.map((d) => d.years[1]));
      return drivers.filter((d) => d.years[1] === latest).sort((a, b) => b.starts - a.starts).slice(0, MAX_CARS);
    }],
    ["Random", () => [...drivers].filter((d) => d.starts >= 30).sort(() => Math.random() - 0.5).slice(0, MAX_CARS)],
  ];

  return (
    <>
      <h1>Every driver. The same car.</h1>
      <p className="lead">
        Pick up to {MAX_CARS} drivers from any era since 1950, choose a circuit and race them in identical
        cars. Each driver races at their peak. The race is simulated from a statistical model of driver
        skill, so every race is different.
      </p>

      <div className="grid-2">
        <div className="card stack">
          <div>
            <h3>Circuit</h3>
            <div className="row">
              <select value={props.circuitId} onChange={(event) => props.setCircuitId(event.target.value)} aria-label="Circuit">
                {[...circuits].sort((a, b) => a.country.localeCompare(b.country)).map((circuit) => (
                  <option key={circuit.id} value={circuit.id}>
                    {circuit.country}: {circuit.name}
                  </option>
                ))}
              </select>
              <select value={props.distance} onChange={(event) => props.setDistance(event.target.value as "sprint")} aria-label="Race distance">
                <option value="sprint">Sprint (15 laps)</option>
                <option value="half">Half distance</option>
                <option value="full">Full distance</option>
              </select>
            </div>
          </div>
          <div>
            <h3>Quick grids</h3>
            <div className="row">
              {presets.map(([label, build]) => (
                <button key={label} className="chip" onClick={() => setEntrants(build())}>
                  {label}
                </button>
              ))}
              <button className="chip" onClick={() => setEntrants([])}>Clear</button>
            </div>
          </div>
          <div>
            <h3>Add a driver</h3>
            <DriverPicker
              drivers={drivers.filter((driver) => !chosen.has(driver.id))}
              onPick={(driver) => !full && setEntrants([...entrants, driver])}
              placeholder={full ? "The grid is full" : `Search all ${drivers.length} drivers since 1950...`}
            />
          </div>
          <button className="primary" disabled={entrants.length < 2} onClick={props.onStart}>
            Start race ({props.laps} laps)
          </button>
        </div>

        <div className="card">
          <h3>
            Grid ({entrants.length}/{MAX_CARS})
          </h3>
          {entrants.length === 0 && <p className="muted">No drivers yet. Pick a quick grid or search.</p>}
          <div className="grid-list">
            {entrants.map((driver, car) => (
              <div className="grid-slot" key={driver.id}>
                <CarBadge car={car} />
                <span>
                  {driver.name} <span className="muted small num">{driver.peak[0]}–{String(driver.peak[1]).slice(2)}</span>
                </span>
                <button aria-label={`Remove ${driver.name}`} onClick={() => setEntrants(entrants.filter((d) => d.id !== driver.id))}>
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ---- The pieces of the race screen ------------------------------------------

function CarBadge({ car }: { car: number }) {
  const look = carLook(car);
  return (
    <span className="badge num" style={{ background: look.fill, boxShadow: `inset 0 0 0 2px ${look.stroke}` }}>
      {car + 1}
    </span>
  );
}

type TrackProps = {
  points: [number, number][];
  entrants: Driver[];
  carElements: React.MutableRefObject<(SVGGElement | null)[]>;
};

function Track({ points, entrants, carElements }: TrackProps) {
  const height = Math.max(...points.map(([, y]) => y));
  const width = Math.max(...points.map(([x]) => x));
  const path = points.map(([x, y]) => `${x},${y}`).join(" ");
  const [startX, startY] = points[0];
  return (
    <svg viewBox={`-50 -50 ${width + 100} ${height + 100}`} role="img" aria-label="Cars moving round the track outline">
      <polygon points={path} className="track-outline" />
      <polygon points={path} className="track-centre" />
      <circle cx={startX} cy={startY} r={9} fill="#ffffff" />
      {entrants.map((driver, car) => {
        const look = carLook(car);
        return (
          <g
            key={driver.id}
            className="car"
            ref={(element) => {
              carElements.current[car] = element;
            }}
            transform={`translate(${startX} ${startY})`}
          >
            <circle r={17} fill={look.fill} stroke={look.stroke} strokeWidth={look.strokeWidth} />
            <text fill="#ffffff">{car + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

function Tower({ rows, entrants, race }: { rows: TowerRow[]; entrants: Driver[]; race: Race }) {
  // Remember each car's last position, to flash a row green or red when it changes.
  const lastPlace = useRef<Record<number, number>>({});
  const flash = useRef<Record<number, { kind: string; until: number }>>({});
  const now = typeof performance === "undefined" ? 0 : performance.now();
  rows.forEach((row, place) => {
    const before = lastPlace.current[row.car];
    if (before !== undefined && before !== place && !row.out) {
      flash.current[row.car] = { kind: place < before ? "gained" : "lost", until: now + 1200 };
    }
    lastPlace.current[row.car] = place;
  });

  return (
    <div className="tower" style={{ height: rows.length * TOWER_ROW_HEIGHT }}>
      {/* Rows are always listed in car order and moved with CSS, so a row
          slides to its new place when the order changes. */}
      {entrants.map((driver, car) => {
        const place = rows.findIndex((row) => row.car === car);
        if (place < 0) return null;
        const row = rows[place];
        const flashing = flash.current[car] && flash.current[car].until > now ? flash.current[car].kind : "";
        return (
          <div
            key={driver.id}
            className={`tower-row ${row.out ? "out" : ""} ${flashing}`}
            style={{ transform: `translateY(${place * TOWER_ROW_HEIGHT}px)` }}
          >
            <span className="tower-pos num">{place + 1}</span>
            <CarBadge car={car} />
            <span className="tower-name">{surname(driver)}</span>
            <span className="tower-gap num">
              {row.out ? "OUT" : place === 0 ? (row.lapsDone >= race.laps ? "Winner" : "Leader") : `+${row.gap.toFixed(3)}`}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function Results({ race, entrants }: { race: Race; entrants: Driver[] }) {
  const winner = race.finishOrder[0];
  const winnerTime = race.lineTimes[winner][race.laps];
  const quickest = Math.min(...race.bestLap);
  return (
    <div className="card">
      <h2>Result: {entrants[winner].name} wins</h2>
      <div className="table-scroll">
        <table className="results">
          <thead>
            <tr>
              <th>Pos</th>
              <th>Driver</th>
              <th>Grid</th>
              <th className="hide-phone">Best lap</th>
              <th>Gap</th>
            </tr>
          </thead>
          <tbody>
            {race.finishOrder.map((car, place) => {
              const crashed = race.crashLap[car] !== null;
              const gained = race.grid[car] - place;
              return (
                <tr key={car}>
                  <td className="num">{place + 1}</td>
                  <td>
                    <span className="row" style={{ flexWrap: "nowrap" }}>
                      <CarBadge car={car} /> {entrants[car].name}
                    </span>
                  </td>
                  <td className="num">
                    {race.grid[car] + 1}{" "}
                    {gained !== 0 && !crashed && (
                      <span className={gained > 0 ? "delta-up" : "delta-down"}>
                        {gained > 0 ? "▲" : "▼"}
                        {Math.abs(gained)}
                      </span>
                    )}
                  </td>
                  <td className="num hide-phone">
                    {Number.isFinite(race.bestLap[car]) ? lapTimeText(race.bestLap[car]) : ""}
                    {race.bestLap[car] === quickest && <span className="muted small"> fastest</span>}
                  </td>
                  <td className="num">
                    {crashed
                      ? `Crashed, lap ${race.crashLap[car]}`
                      : place === 0
                        ? lapTimeText(winnerTime)
                        : `+${(race.lineTimes[car][race.laps] - winnerTime).toFixed(3)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

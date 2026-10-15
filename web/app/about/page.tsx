// app/about/page.tsx
//
// What it does: the about page. A short, honest explanation of the method and
//   of every test the model was put through, with the numbers read straight
//   from the files the Python scripts wrote, so the page can never disagree
//   with the real results:
//     validation.json   the 2024 and 2025 test (validate_model.py)
//     backtest.json     every season 2010 to 2025 predicted from earlier ones (backtest.py)
//     sensitivity.json  how the top 20 moves under other assumptions (sensitivity.py)
//     improvements.json how far prediction can go (backtest_improvements.py)
// What it reads: those four files from public/data.
// Which files use it: none (it is a page).

"use client";

import { useState } from "react";

import { useData } from "@/lib/data";

type TeammateScore = { comparisons: number; model: number; career_points: number; last_season_or_points: number };
type CalibrationRow = { model_said: string; comparisons: number; "average_said_%": number; "favourite_won_%": number };
type Validation = {
  trainedUntil: number;
  temperature: number;
  teammates: Record<string, TeammateScore>;
  calibration: { before: CalibrationRow[]; after: CalibrationRow[] };
  probabilityScores: { before: { brier: number; log_loss: number }; after: { brier: number; log_loss: number } };
  positions: Record<string, Record<string, { average_miss_places: number }>>;
};
type SeasonScore = { comparisons: number; model: number; career_points: number; last_season_or_points: number; model_log_loss: number; model_brier: number };
type Backtest = {
  temperature: number;
  seasons: { season: number; race: SeasonScore; quali: SeasonScore }[];
  summary: { race: SeasonScore; quali: SeasonScore };
  summaryUncalibrated: { race: SeasonScore };
  variants: Record<string, { race: SeasonScore }>;
};
type ImprovementRow = {
  label: string;
  race: { accuracy: number; brier: number; log_loss: number };
  quali: { accuracy: number; brier: number; log_loss: number } | null; // null: not applicable
};
type Improvements = { firstSeason: number; lastSeason: number; table: ImprovementRow[] };
type Sensitivity = {
  assumptions: Record<string, number>;
  solidWithin: number;
  top: { id: string; name: string; ranks: Record<string, number>; solid: boolean }[];
  onlyUnderOther: Record<string, string[]>;
};

const percent = (share: number) => `${(share * 100).toFixed(1)}%`;

export default function AboutPage() {
  const validation = useData<Validation>("validation.json");
  const backtest = useData<Backtest>("backtest.json");
  const sensitivity = useData<Sensitivity>("sensitivity.json");
  const improvements = useData<Improvements>("improvements.json");

  return (
    <div className="prose">
      <h1>How this works</h1>
      <p className="lead">
        Formula 1 results mix two things: how good the driver is and how good the car is. This project
        tries to separate them, and is honest about how far that can be done.
      </p>

      <h2>The idea</h2>
      <p>
        Teammates drive the same car. So when one teammate is consistently quicker than the other, that
        is the driver, not the car. The model looks at every pair of teammates in every race since 1950:
        who finished ahead, the gap between their qualifying laps (from 1994) and the gap in their race
        pace (from 2018).
      </p>
      <p>
        Drivers change teams, so teammates link up into chains. Senna raced alongside Prost, Prost
        alongside Lauda, and so on back to 1950. Through those chains, two drivers who never met can
        still be compared. 643 drivers are connected this way. Each driver card shows the chain that
        links that driver to today&apos;s grid, and the teammate battles that moved their rating most.
      </p>
      <p>
        Races that ended with a car failure are left out, so a breakdown never counts against a driver.
        A driver&apos;s own crash does count.
      </p>

      <h2>What the numbers mean</h2>
      <ul>
        <li>
          <b>Skill</b> is in percent of lap time. +0.5% means half a percent quicker than an average
          newcomer of today in the same car: about 0.45 seconds on a 90 second lap.
        </li>
        <li>
          <b>The range</b> is where the true value probably lies (9 times in 10, if the model&apos;s
          assumptions hold). It comes from 1,000 plausible versions of every rating, drawn by a sampler
          straight from the model, so ratings that move together (old teammates) are handled properly.
          Drivers with few races, few teammates or from older eras have wider ranges.
        </li>
        <li>
          <b>Rank</b> is only given to drivers with at least 30 starts, and &quot;could rank&quot; is the middle
          90% of their plausible ranks among those drivers. A driver with three races has a rating that
          is mostly guesswork; ranking everyone against such drivers made every range look wider than
          it is.
        </li>
        <li>
          <b>Peak</b> is a driver&apos;s best three seasons in a row. Rankings and races use the peak.
        </li>
        <li>
          <b>Consistency</b> (50 is typical) combines how rarely a driver crashed out compared with their
          era and, from 2018, how steady their lap times were.
        </li>
        <li>
          <b>Wet weather</b> is a separate, exploratory estimate. Only 54 races since 1950 are tagged as
          wet (37 from a hand-made list of well known wet races before 2018, 17 measured from lap data
          since), so most drivers&apos; wet adjustments rest on a handful of races and stay close to zero.
          Adding it to the main model made predictions slightly worse, so the main ratings do not use it.
        </li>
      </ul>

      <h2>Was it tested?</h2>
      {backtest && validation ? (
        <>
          <p>
            Two ways. First, a rolling backtest: for every season from 2010 to 2025, the model was fitted
            on all earlier seasons only and asked which teammate would finish ahead in that season.
          </p>
          <BacktestChart backtest={backtest} />
          <div className="card table-scroll" style={{ margin: "12px 0" }}>
            <h3>All {backtest.seasons.length} seasons pooled ({backtest.summary.race.comparisons} race comparisons)</h3>
            <table>
              <tbody>
                <tr><td>Model</td><td className="right num"><b>{percent(backtest.summary.race.model)}</b></td></tr>
                <tr><td>Last season&apos;s head to head repeats, otherwise more career points</td><td className="right num">{percent(backtest.summary.race.last_season_or_points)}</td></tr>
                <tr><td>More career points</td><td className="right num">{percent(backtest.summary.race.career_points)}</td></tr>
              </tbody>
            </table>
          </div>
          <p>
            Honestly put: over sixteen seasons the model is only slightly better than simply assuming last
            season&apos;s teammate battle repeats. Its real advantage shows when drivers change teams or
            are new, as in 2025, because it can compare drivers who were never teammates.
          </p>

          <p>
            Second, a single test: fitted up to the end of {validation.trainedUntil}, then predicting 2024 and
            2025 without being refitted.
          </p>
          <div className="card table-scroll" style={{ margin: "12px 0" }}>
            <h3>Which teammate finishes ahead? Share called correctly</h3>
            <table>
              <thead>
                <tr>
                  <th>Test</th>
                  <th className="right">Comparisons</th>
                  <th className="right">Model</th>
                  <th className="right">More career points</th>
                  <th className="right">Last season repeats</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(validation.teammates).map(([test, score]) => (
                  <tr key={test}>
                    <td>{test}</td>
                    <td className="right num">{score.comparisons}</td>
                    <td className="right num"><b>{percent(score.model)}</b></td>
                    <td className="right num">{percent(score.career_points)}</td>
                    <td className="right num">{percent(score.last_season_or_points)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2>Are its probabilities honest?</h2>
          <p>
            The raw model turned out too confident when predicting seasons it had not seen: across 2010 to
            2023 its favourites won less often than it said. So every probability on this site is softened
            by one learned number (a &quot;temperature&quot; of {backtest.temperature}, fitted on all
            the backtest&apos;s predictions). The catch, stated plainly: the right amount of softening has
            changed over time. Older seasons needed a lot; seasons since 2018 needed almost none. On the 2024
            and 2025 test, softening by the amount learned from 2010 to 2023 made the probabilities worse:
          </p>
          <div className="card table-scroll" style={{ margin: "12px 0" }}>
            <h3>2024 and 2025 races, before and after softening (lower is better)</h3>
            <table>
              <thead><tr><th></th><th className="right">Brier score</th><th className="right">Log loss</th></tr></thead>
              <tbody>
                <tr><td>Before</td><td className="right num">{validation.probabilityScores.before.brier.toFixed(4)}</td><td className="right num">{validation.probabilityScores.before.log_loss.toFixed(4)}</td></tr>
                <tr><td>After (temperature {validation.temperature})</td><td className="right num">{validation.probabilityScores.after.brier.toFixed(4)}</td><td className="right num">{validation.probabilityScores.after.log_loss.toFixed(4)}</td></tr>
                <tr><td className="muted">A coin flip</td><td className="right num muted">0.2500</td><td className="right num muted">0.6931</td></tr>
              </tbody>
            </table>
          </div>
          <p className="muted small">
            So probabilities for modern drivers on this site are probably too cautious, and those for older
            drivers about right. The choice was made by the rule set in advance (fit on earlier seasons), not
            changed after seeing the test.
          </p>
        </>
      ) : (
        <p className="muted">Loading results...</p>
      )}

      <h2>How far can prediction go?</h2>
      <p>
        Even knowing a season&apos;s results in hindsight, a model can only call so many teammate battles,
        because single races are full of luck. The table compares the current model with three possible
        improvements (refitting before every race, using the same weekend&apos;s qualifying, and a curve for
        how drivers improve with experience), and with two hindsight references that are allowed to see
        the answers, so they show how much room there is. Every fair row predicts each race only from
        earlier races.
      </p>
      {improvements ? <ImprovementTable improvements={improvements} /> : <p className="muted">Loading...</p>}

      <h2>Which rankings depend on an assumption?</h2>
      <p>
        One thing the data cannot settle: did the general level of F1 drivers rise or fall between decades?
        The model assumes it changed a little. Here is the top 20 under three different assumptions.
      </p>
      {sensitivity ? <SensitivityTable sensitivity={sensitivity} /> : <p className="muted">Loading...</p>}

      <h2>What to be sceptical about</h2>
      <p className="note">
        The races on this site are simulated from a model. They are not real results, and they are not
        predictions of what would really have happened.
      </p>
      <ul>
        <li>
          <b>Comparisons across eras cannot be tested.</b> There is no way to check Fangio against
          Verstappen. Only the modern end of the model has been validated.
        </li>
        <li>
          <b>A driver is only ever measured against their teammates.</b> Someone whose teammates were all
          weak, or who was a clear team number one, can be flattered. Two fixes were tried (leaving out
          races arranged by documented team orders, and assuming short-career teammates were weaker); neither
          improved the predictions, so neither is used.
        </li>
        <li>
          <b>From 2023 the data no longer says why a car retired</b>, so those races are left out and a
          driver&apos;s own crashes stop counting against them from then on.
        </li>
        <li>
          <b>Before about 1970, &quot;teammates&quot; are sometimes private owners of the same make of car</b>,
          whose cars were not equal.
        </li>
        <li>
          <b>The race simulation leaves out</b> tyres, pit stops, safety cars and weather. Its randomness is
          tuned so that drivers beat each other as often as the model says, and overtaking difficulty for
          each circuit is measured from real races since 2018.
        </li>
        <li>
          <b>The Pit Wall assistant</b> answers only from these numbers and checks every number it writes
          against them, but it is a language model and can still word things badly.
        </li>
      </ul>

      <h2>Data and code</h2>
      <p>
        Results and qualifying since 1950 come from the Jolpica F1 API. Lap times and track outlines from
        2018 come from FastF1. Both are public. No team or series logos or images are used. This site is
        not affiliated with Formula 1.
      </p>
    </div>
  );
}

/** Share called correctly per season: model against the two baselines, with hover values. */
function BacktestChart({ backtest }: { backtest: Backtest }) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 640, height = 220, left = 40, right = 10, top = 10, bottom = 26;
  const seasons = backtest.seasons;
  const x = (i: number) => left + (i / (seasons.length - 1)) * (width - left - right);
  const y = (share: number) => top + ((0.8 - share) / 0.4) * (height - top - bottom); // 40% to 80%
  const series = [
    { key: "model" as const, label: "Model", colour: "var(--series-1)", widthPx: 2.5 },
    { key: "last_season_or_points" as const, label: "Last season repeats", colour: "var(--series-2)", widthPx: 2 },
    { key: "career_points" as const, label: "More career points", colour: "var(--muted)", widthPx: 2 },
  ];
  const shown = hover === null ? null : seasons[hover];
  return (
    <div className="card chart-wrap" style={{ margin: "12px 0" }}>
      <h3>Teammate race results called correctly, each season predicted from earlier seasons only</h3>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}
        role="img" aria-label="Accuracy by season for the model and two baselines"
        onMouseMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const at = ((event.clientX - box.left) / box.width) * width;
          setHover(Math.max(0, Math.min(seasons.length - 1, Math.round(((at - left) / (width - left - right)) * (seasons.length - 1)))));
        }}
        onMouseLeave={() => setHover(null)}>
        {[0.4, 0.5, 0.6, 0.7, 0.8].map((share) => (
          <g key={share}>
            <line x1={left} x2={width - right} y1={y(share)} y2={y(share)} stroke={share === 0.5 ? "var(--line-strong)" : "var(--line)"} />
            <text x={left - 6} y={y(share) + 4} textAnchor="end" className="axis-text">{share * 100}%</text>
          </g>
        ))}
        {seasons.map((row, i) => (i % 3 === 0 || i === seasons.length - 1) && (
          <text key={row.season} x={x(i)} y={height - 6} textAnchor="middle" className="axis-text">{row.season}</text>
        ))}
        {series.map((s) => (
          <polyline key={s.key} fill="none" stroke={s.colour} strokeWidth={s.widthPx} strokeLinejoin="round"
            points={seasons.map((row, i) => `${x(i)},${y(row.race[s.key])}`).join(" ")} />
        ))}
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={top} y2={height - bottom} stroke="var(--muted)" />}
      </svg>
      {shown && (
        <div className="tooltip num" style={{ left: `${(x(hover!) / width) * 100}%`, top: "30%" }}>
          <b>{shown.season}</b> ({shown.race.comparisons} pairs): model {percent(shown.race.model)}, last season{" "}
          {percent(shown.race.last_season_or_points)}, career points {percent(shown.race.career_points)}
        </div>
      )}
      <div className="legend">
        {series.map((s) => <span key={s.key}><i style={{ background: s.colour }} />{s.label}</span>)}
      </div>
    </div>
  );
}

function ImprovementTable({ improvements }: { improvements: Improvements }) {
  const cell = (value: number) => <td className="right num">{value.toFixed(4)}</td>;
  return (
    <div className="card table-scroll" style={{ margin: "12px 0" }}>
      <h3>Teammate predictions, {improvements.firstSeason} to {improvements.lastSeason} pooled</h3>
      <table>
        <thead>
          <tr>
            <th></th>
            <th className="right">Race</th><th className="right">Brier</th><th className="right">Log loss</th>
            <th className="right">Qualifying</th><th className="right">Brier</th><th className="right">Log loss</th>
          </tr>
        </thead>
        <tbody>
          {improvements.table.map((row) => (
            <tr key={row.label} className={row.label.includes("indsight") ? "muted" : ""}>
              <td>{row.label}</td>
              <td className="right num">{percent(row.race.accuracy)}</td>{cell(row.race.brier)}{cell(row.race.log_loss)}
              {row.quali ? (
                <><td className="right num">{percent(row.quali.accuracy)}</td>{cell(row.quali.brier)}{cell(row.quali.log_loss)}</>
              ) : (
                <td className="right muted small" colSpan={3}>not applicable</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        Lower Brier and log loss are better; a coin flip scores 0.25 and 0.693. The greyed rows know the
        answers and are not fair models.
      </p>
    </div>
  );
}

function SensitivityTable({ sensitivity }: { sensitivity: Sensitivity }) {
  const labels = Object.keys(sensitivity.assumptions);
  const others = Object.entries(sensitivity.onlyUnderOther).filter(([, names]) => names.length > 0);
  return (
    <>
      <div className="card table-scroll" style={{ margin: "12px 0" }}>
        <table>
          <thead>
            <tr>
              <th>Driver</th>
              {labels.map((label) => (
                <th key={label} className="right">{label}<br /><span className="muted">{sensitivity.assumptions[label]}% a decade</span></th>
              ))}
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {sensitivity.top.map((row) => (
              <tr key={row.id}>
                <td style={{ fontWeight: 600 }}>{row.name}</td>
                {labels.map((label) => <td key={label} className="right num">{row.ranks[label]}</td>)}
                <td className={row.solid ? "" : "muted"}>{row.solid ? "solid" : "depends on the assumption"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        &quot;Solid&quot; means the rank moves by at most {sensitivity.solidWithin} places across all three assumptions.
        {others.map(([label, names]) => ` Only in the top 20 under "${label}": ${names.join(", ")}.`)}
      </p>
    </>
  );
}

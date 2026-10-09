// app/about/page.tsx
//
// What it does: the about page. A short, honest explanation of the method,
//   the validation results (read from the file validate_model.py wrote, so
//   they can never disagree with the real numbers) and the limitations.
// What it reads: validation.json from public/data.
// Which files use it: none (it is a page).

"use client";

import { useData } from "@/lib/data";

type TeammateScore = { comparisons: number; model: number; career_points: number; last_season_or_points: number };
type Validation = {
  trainedUntil: number;
  testYears: number[];
  teammates: Record<string, TeammateScore>;
  calibration: Record<string, { model_said: string; comparisons: number; "average_said_%": number; "favourite_won_%": number }[]>;
  positions: Record<string, Record<string, { average_miss_places: number }>>;
};

const percent = (share: number) => `${(share * 100).toFixed(1)}%`;

export default function AboutPage() {
  const validation = useData<Validation>("validation.json");

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
        alongside Lauda, Lauda alongside Regazzoni, and so on back to 1950. Through those chains, two
        drivers who never met can still be compared. 643 drivers are connected this way.
      </p>
      <p>
        Races that ended with a car failure are left out, so a breakdown never counts against a driver.
        A driver's own crash does count.
      </p>

      <h2>What the numbers mean</h2>
      <ul>
        <li>
          <b>Skill</b> is in percent of lap time. +0.5% means half a percent quicker than an average
          newcomer of today in the same car: about 0.45 seconds on a 90 second lap.
        </li>
        <li>
          <b>The range</b> is where the true value probably lies (9 times in 10, if the model's
          assumptions hold). Drivers with few races, few teammates or from older eras have wider ranges.
        </li>
        <li>
          <b>Peak</b> is a driver's best three seasons in a row. Rankings and races use the peak.
        </li>
        <li>
          <b>Consistency</b> (50 is typical) combines how rarely a driver crashed out compared with their
          era and, from 2018, how steady their lap times were.
        </li>
      </ul>

      <h2>Was it tested?</h2>
      <p>
        Yes. The model was fitted on results up to the end of {validation?.trainedUntil ?? 2023} and then asked
        to predict 2024 and 2025, which it had never seen, without being refitted.
      </p>
      {validation ? (
        <>
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
          <p>
            In 2024 the model only matched the simple baselines: nearly every team kept the same two
            drivers as 2023, so "last season repeats" was hard to beat. In 2025 many drivers moved or were
            new, the baselines fell, and the model held up. A coin flip scores 50%.
          </p>
          <div className="card table-scroll" style={{ margin: "12px 0" }}>
            <h3>Are its probabilities honest? Races, 2024 and 2025</h3>
            <table>
              <thead>
                <tr>
                  <th>Model said the favourite had</th>
                  <th className="right">Comparisons</th>
                  <th className="right">Average said</th>
                  <th className="right">Favourite really won</th>
                </tr>
              </thead>
              <tbody>
                {validation.calibration.race.map((row) => (
                  <tr key={row.model_said}>
                    <td>{row.model_said}</td>
                    <td className="right num">{row.comparisons}</td>
                    <td className="right num">{row["average_said_%"]}%</td>
                    <td className="right num">{row["favourite_won_%"]}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>The model is too cautious about clear favourites: they won more often than it said.</p>
          <div className="card table-scroll" style={{ margin: "12px 0" }}>
            <h3>Finishing position, predicted from 2023 only. Average miss in places</h3>
            <table>
              <thead>
                <tr>
                  <th>Season</th>
                  {Object.keys(Object.values(validation.positions)[0]).map((method) => (
                    <th key={method} className="right">{method}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(validation.positions).map(([year, methods]) => (
                  <tr key={year}>
                    <td>{year}</td>
                    {Object.values(methods).map((score, i) => (
                      <td key={i} className="right num">{score.average_miss_places.toFixed(2)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Knowing the driver as well as the car helps. But for 2025 the model is worse than simply
            using the 2024 standings, because it was still using 2023 cars.
          </p>
        </>
      ) : (
        <p className="muted">Loading results...</p>
      )}

      <h2>What to be sceptical about</h2>
      <p className="note">
        The races on this site are simulated from a model. They are not real results, and they are not
        predictions of what would really have happened.
      </p>
      <ul>
        <li>
          <b>Comparisons across eras cannot be tested.</b> There is no way to check Fangio against
          Verstappen. Only the modern end of the model has been validated. That is why the ranges for
          older drivers are wide, and why most "could rank" ranges span dozens of places.
        </li>
        <li>
          <b>How much the general level of drivers changed between decades is an assumption</b>, not a
          measurement. A different assumption would move every older driver's range.
        </li>
        <li>
          <b>A driver is only ever measured against their teammates.</b> Someone whose teammates were all
          weak, or who was a clear team number one, can be flattered. Someone hired to support a
          champion can be undersold.
        </li>
        <li>
          <b>From 2023 the data no longer says why a car retired</b>, so those races are left out and a
          driver's own crashes stop counting against them from then on.
        </li>
        <li>
          <b>Before about 1970, "teammates" are sometimes private owners of the same make of car</b>, whose
          cars were not equal.
        </li>
        <li>
          <b>The race simulation leaves out</b> tyres, pit stops, safety cars and weather. Its randomness is
          tuned so that drivers beat each other about as often as the tested model says they should.
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

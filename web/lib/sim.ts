// lib/sim.ts
//
// What it does: simulates one equal car race in the browser. It is the same
//   race as simulate.py (the GPU version), rewritten for a single race, and it
//   uses the same numbers: they arrive in settings.json, so the two cannot
//   drift apart.
// What it reads: nothing. The race page passes in the drivers and settings.
// What it produces: a Race: every car's time at the end of every lap, which is
//   all the animation and the timing tower need.
// Which files use it: app/page.tsx (the fantasy race).

import type { SimSettings } from "./data";

export type Entrant = { skill: number; sd: number; consistency: number };

export type Race = {
  laps: number;
  // lineTimes[car][lap] = race time when the car completes that lap.
  // lineTimes[car][0] is when it leaves the grid (pole leaves at 0).
  lineTimes: number[][];
  grid: number[]; // grid slot of each car, 0 = pole
  crashLap: (number | null)[]; // the lap a car crashed on, or null
  bestLap: number[]; // each car's quickest lap in seconds
  finishOrder: number[]; // car numbers, winner first
};

/** A random number from a bell curve centred on 0 (the Box-Muller method). */
function bell(): number {
  const a = 1 - Math.random(); // 1 - x avoids log(0)
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * Math.random());
}

/** 1 for a typical driver, 0.5 for a consistency of 75, 2 for 25. Same as simulate.py. */
function errorFactor(consistency: number): number {
  return 2 ** ((50 - consistency) / 25);
}

export function passMargin(difficulty: number, s: SimSettings): number {
  return s.passMarginEasy + difficulty * (s.passMarginHard - s.passMarginEasy);
}

export function simulateRace(
  entrants: Entrant[], lapSeconds: number, laps: number, difficulty: number, s: SimSettings,
): Race {
  const count = entrants.length;
  const cars = entrants.map((_, car) => car);
  const percent = lapSeconds / 100; // one percent of a lap, in seconds
  const margin = passMargin(difficulty, s);

  // Each race draws a plausible version of every rating (it is uncertain)
  // and a form for the day. This is why no two races are the same.
  const pace = entrants.map((e) => lapSeconds - (e.skill + bell() * e.sd + bell() * s.formSd) * percent);
  const errors = entrants.map((e) => errorFactor(e.consistency));

  // Qualifying: one lap each. The quickest starts first.
  const qualiLap = pace.map((p) => p + bell() * s.qualiNoise * percent);
  const qualiOrder = [...cars].sort((a, b) => qualiLap[a] - qualiLap[b]);
  const grid = new Array<number>(count);
  qualiOrder.forEach((car, slot) => (grid[car] = slot));

  const raceTime = grid.map((slot) => slot * s.gridGapSeconds);
  const lineTimes = raceTime.map((time) => [time]);
  const crashLap: (number | null)[] = new Array(count).fill(null);
  const bestLap = new Array<number>(count).fill(Infinity);

  for (let lap = 1; lap <= laps; lap++) {
    const lapTime = cars.map((car) => {
      const mistake = Math.random() < s.mistakeChance * errors[car]
        ? -Math.log(1 - Math.random()) * s.mistakeSeconds : 0;
      return pace[car] + bell() * s.lapNoise * percent * errors[car] + mistake;
    });

    // Take the cars in the order they started the lap. A car only gets past
    // the one in front if it was quicker by more than the passing margin;
    // otherwise it ends the lap stuck just behind (see hold_up in simulate.py).
    const running = cars.filter((car) => crashLap[car] === null).sort((a, b) => raceTime[a] - raceTime[b]);
    let ahead = -1;
    for (const car of running) {
      let finish = raceTime[car] + lapTime[car];
      if (ahead >= 0) {
        const wouldPass = finish < raceTime[ahead] + s.followGapSeconds;
        const fastEnough = lapTime[ahead] - lapTime[car] > margin;
        // A stuck car follows between one and two following gaps behind.
        if (wouldPass && !fastEnough) finish = raceTime[ahead] + s.followGapSeconds * (1 + Math.random());
      }
      bestLap[car] = Math.min(bestLap[car], finish - raceTime[car]);
      raceTime[car] = finish;
      lineTimes[car].push(finish);
      ahead = car;
    }

    for (const car of running) {
      if (Math.random() < (s.crashChance / laps) * errors[car]) crashLap[car] = lap;
    }
  }

  // Finishers by race time, then crashed cars: the later the crash, the higher.
  const finishOrder = [...cars].sort((a, b) => {
    const [crashA, crashB] = [crashLap[a], crashLap[b]];
    if (crashA === null && crashB === null) return raceTime[a] - raceTime[b];
    if (crashA === null) return -1;
    if (crashB === null) return 1;
    return crashB - crashA;
  });
  return { laps, lineTimes, grid, crashLap, bestLap, finishOrder };
}

/**
 * How far a car has travelled at a moment of the race, in laps (12.37 = 37%
 * of the way round lap 13). Between two crossings of the line the car is
 * assumed to move at a steady speed.
 */
export function progressAt(race: Race, car: number, time: number): number {
  const times = race.lineTimes[car];
  if (time <= times[0]) return 0;
  const last = times.length - 1;
  if (time >= times[last]) return last;
  let lap = 0;
  while (times[lap + 1] <= time) lap++;
  return lap + (time - times[lap]) / (times[lap + 1] - times[lap]);
}

/** Has this car crashed out by this moment? (It stops where the crash lap ended.) */
export function isOut(race: Race, car: number, time: number): boolean {
  const lap = race.crashLap[car];
  return lap !== null && time >= race.lineTimes[car][lap];
}

/** The moment the last running car takes the flag. */
export function raceLength(race: Race): number {
  return Math.max(...race.lineTimes.map((times, car) => (race.crashLap[car] === null ? times[race.laps] : 0)));
}

export type TowerRow = { car: number; out: boolean; lapsDone: number; gap: number };

/**
 * The timing tower at a moment of the race: cars in race order with the gap
 * to the leader. Like real F1 timing, a gap is measured where the car last
 * crossed the line: its time there minus the leader's time at the same line.
 */
export function towerAt(race: Race, time: number): TowerRow[] {
  const rows = race.lineTimes.map((_, car) => ({
    car, out: isOut(race, car, time), progress: progressAt(race, car, time),
  }));
  rows.sort((a, b) => {
    if (a.out !== b.out) return a.out ? 1 : -1;
    if (a.out) return (race.crashLap[b.car] as number) - (race.crashLap[a.car] as number);
    // Level on distance (both on the grid, or both finished): earlier time wins.
    const lapA = Math.floor(a.progress), lapB = Math.floor(b.progress);
    return b.progress - a.progress || race.lineTimes[a.car][lapA] - race.lineTimes[b.car][lapB];
  });
  const leader = rows[0].car;
  return rows.map((row) => {
    const lapsDone = Math.floor(row.progress);
    const gap = race.lineTimes[row.car][lapsDone] - race.lineTimes[leader][lapsDone];
    return { car: row.car, out: row.out, lapsDone, gap };
  });
}

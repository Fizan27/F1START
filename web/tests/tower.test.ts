// tests/tower.test.ts
//
// What it does: checks the timing tower order (towerAt and placeChanges in
//   lib/sim.ts) on a small hand-made race where the right order is known at
//   every moment: the grid order before the start, the race order during the
//   race (including after an overtake), crashed cars at the bottom, and the
//   finishing order at the flag.
// Which files use it: none.  Run:  cd web ; npm test

import { describe, expect, it } from "vitest";

import { placeChanges, type Race, towerAt } from "@/lib/sim";

// Three cars, three laps. Car 1 starts on pole, car 2 second, car 0 third.
// Car 0 passes both on lap 1 and wins. Car 2 crashes on lap 2.
const race: Race = {
  laps: 3,
  lineTimes: [
    [0.5, 90.5, 181.0, 271.6], // car 0
    [0.0, 90.6, 182.0, 272.0], // car 1
    [0.25, 91.0, 182.5], // car 2: crashed at the end of lap 2
  ],
  grid: [2, 0, 1],
  gridFromQualifying: true,
  crashLap: [null, null, 2],
  bestLap: [90.0, 90.6, 90.75],
  finishOrder: [0, 1, 2],
  safetyCarLap: null,
};

const order = (time: number) => towerAt(race, time).map((row) => row.car);

describe("towerAt", () => {
  it("shows the grid order before the start, pole at the top", () => {
    expect(order(0)).toEqual([1, 2, 0]);
  });

  it("lists cars by race position, P1 first, and updates after an overtake", () => {
    expect(order(10)).toEqual([1, 2, 0]); // early on lap 1, still in grid order
    expect(order(45)).toEqual([1, 0, 2]); // halfway round, car 0 (quicker on this lap) has passed car 2
    expect(order(100)).toEqual([0, 1, 2]); // car 0 crossed the line first at the end of lap 1
  });

  it("puts a crashed car at the bottom, marked out", () => {
    const rows = towerAt(race, 200);
    expect(rows.map((row) => row.car)).toEqual([0, 1, 2]);
    expect(rows[2].out).toBe(true);
    expect(rows[0].out).toBe(false);
  });

  it("gives the gap to the leader where each car last crossed the line", () => {
    const rows = towerAt(race, 100);
    expect(rows[0].gap).toBe(0);
    expect(rows[1].gap).toBeCloseTo(0.1, 6); // 90.6 - 90.5 at the end of lap 1
  });

  it("shows the finishing order at the flag", () => {
    expect(order(272)).toEqual([0, 1, 2]);
  });
});

describe("placeChanges", () => {
  it("reports places gained as positive and places lost as negative", () => {
    const before = [1, 2, 0];
    const after = towerAt(race, 100); // car 0 from 3rd to 1st
    expect(placeChanges(before, after)).toEqual({ 0: 2, 1: -1, 2: -1 });
  });

  it("reports nothing on the first update or when nobody moved", () => {
    expect(placeChanges([], towerAt(race, 0))).toEqual({});
    expect(placeChanges([1, 2, 0], towerAt(race, 0))).toEqual({});
  });
});

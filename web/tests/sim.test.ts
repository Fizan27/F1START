// tests/sim.test.ts
//
// What it does: checks the browser race simulation in lib/sim.ts: a custom
//   grid is used as given, every race gives each place once, a clearly faster
//   driver usually wins, and driver codes are unique.
// Which files use it: none.  Run:  cd web ; npm test

import { describe, expect, it } from "vitest";

import { MODEL_DATA } from "@/lib/modelData";
import { driverCodes, simulateRace } from "@/lib/sim";

const settings = MODEL_DATA.settings.sim;
const entrant = (skill: number) => ({ skill, sd: 0.1, consistency: 50 });

describe("simulateRace", () => {
  it("uses a grid set by the user, and says so", () => {
    const race = simulateRace([entrant(0), entrant(0.5), entrant(0.2)], 90, 10, 0.5, settings, { grid: [2, 0, 1] });
    expect(race.grid).toEqual([2, 0, 1]);
    expect(race.gridFromQualifying).toBe(false);
  });

  it("gives every place exactly once", () => {
    const race = simulateRace([0, 0.1, 0.2, 0.3, 0.4].map(entrant), 90, 20, 0.5, settings);
    expect([...race.finishOrder].sort()).toEqual([0, 1, 2, 3, 4]);
  });

  it("lets a clearly faster driver win most races even from the back of the grid", () => {
    let wins = 0;
    for (let i = 0; i < 200; i++) {
      const race = simulateRace([entrant(1.0), entrant(0), entrant(0), entrant(0)], 90, 50, 0.4, settings, { grid: [3, 0, 1, 2] });
      if (race.finishOrder[0] === 0) wins++;
    }
    expect(wins / 200).toBeGreaterThan(0.6);
  });
});

describe("driverCodes", () => {
  it("makes three letter codes, unique even for drivers with the same surname", () => {
    expect(driverCodes(["Ayrton Senna", "Max Verstappen"])).toEqual(["SEN", "VER"]);
    const codes = driverCodes(["Michael Schumacher", "Ralf Schumacher"]);
    expect(new Set(codes).size).toBe(2);
  });
});

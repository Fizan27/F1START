// tests/tools.test.ts
//
// What it does: checks the assistant's tools against the real exported data:
//   they find drivers the way fans type names, return the model's numbers,
//   and answer unknown drivers with an error instead of a made-up value.
// Which files use it: none.  Run:  cd web ; npm test

import { describe, expect, it } from "vitest";

import { headToHead } from "@/lib/data";
import { MODEL_DATA } from "@/lib/modelData";
import { findDriver, runTool, TOOL_DEFINITIONS } from "@/lib/tools";

describe("findDriver", () => {
  it("finds drivers by surname, full name and without accents", () => {
    expect(findDriver(MODEL_DATA, "Senna")?.id).toBe("senna");
    expect(findDriver(MODEL_DATA, "Lewis Hamilton")?.id).toBe("hamilton");
    expect(findDriver(MODEL_DATA, "Hakkinen")?.id).toBe("hakkinen");
  });
  it("picks the better known driver when a surname is shared", () => {
    expect(findDriver(MODEL_DATA, "Schumacher")?.id).toBe("michael_schumacher");
    expect(findDriver(MODEL_DATA, "Mick Schumacher")?.id).toBe("mick_schumacher");
  });
  it("does not invent a match for a made-up driver", () => {
    expect(findDriver(MODEL_DATA, "Lars Vandersteen")).toBeNull();
    expect(findDriver(MODEL_DATA, "Fernando Verstappen")).toBeNull();
  });
});

describe("tools", () => {
  it("get_driver_rating returns the exported numbers", () => {
    const senna = MODEL_DATA.drivers.find((d) => d.id === "senna")!;
    const result = runTool(MODEL_DATA, "get_driver_rating", { name: "Senna" });
    expect(result.wins).toBe(41);
    expect(result.peakSkillPercent).toBeCloseTo(senna.skill, 2);
    expect(result.rangeLevel).toBe(90);
  });

  it("compare_drivers matches the precomputed head to head and mirrors", () => {
    const stored = headToHead(MODEL_DATA.h2h, "senna", "prost")!;
    const ab = runTool(MODEL_DATA, "compare_drivers", { driver_a: "Senna", driver_b: "Prost" });
    const ba = runTool(MODEL_DATA, "compare_drivers", { driver_a: "Prost", driver_b: "Senna" });
    expect(ab.chanceAFinishesAheadPercent).toBe(stored.best);
    expect(ba.chanceAFinishesAheadPercent).toBe(100 - stored.best);
  });

  it("unknown drivers and unknown tools come back as errors, not numbers", () => {
    expect(runTool(MODEL_DATA, "get_driver_rating", { name: "Lars Vandersteen" })).toHaveProperty("error");
    expect(runTool(MODEL_DATA, "compare_drivers", { driver_a: "Senna", driver_b: "Nobody Atall" })).toHaveProperty("error");
    expect(runTool(MODEL_DATA, "make_tea", {})).toHaveProperty("error");
    expect(runTool(MODEL_DATA, "get_driver_rating", null)).toHaveProperty("error");
  });

  it("get_teammate_record gives real results for real teammates", () => {
    const result = runTool(MODEL_DATA, "get_teammate_record", { driver_a: "Senna", driver_b: "Prost" });
    expect(result.wereTeammates).toBe(true);
    expect(Number(result.raceFinishedAheadA) + Number(result.raceFinishedAheadB)).toBe(result.racesCompared);
  });

  it("simulate_race returns win chances that add up to about 100%", () => {
    const result = runTool(MODEL_DATA, "simulate_race", { drivers: ["Senna", "Prost", "Mansell"], circuit: "Monaco", races: 60 });
    const results = result.results as { winPercent: number }[];
    const total = results.reduce((sum, row) => sum + row.winPercent, 0);
    expect(total).toBeGreaterThanOrEqual(98);
    expect(total).toBeLessThanOrEqual(102);
    expect(result.circuit).toMatch(/Monaco/);
  });

  it("explain_rating and get_wet_rating work for a well known driver, and the wet list is non-empty", () => {
    expect(runTool(MODEL_DATA, "explain_rating", { name: "Hamilton" })).toHaveProperty("pushedRatingUp");
    expect(runTool(MODEL_DATA, "get_wet_rating", { name: "Senna" })).toHaveProperty("wetAdjustmentPercent");
    expect((runTool(MODEL_DATA, "get_wet_rating", {}).bestInTheWet as unknown[]).length).toBeGreaterThan(0);
  });

  it("every tool definition has a schema the API accepts", () => {
    for (const tool of TOOL_DEFINITIONS) {
      expect(tool.input_schema.type).toBe("object");
      expect(tool.description.length).toBeGreaterThan(20);
    }
  });
});

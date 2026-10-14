// tests/numbers.test.ts
//
// What it does: checks the number tracing in lib/numbers.ts, the safety net
//   that catches an assistant answer containing a number no tool returned.
// Which files use it: none.  Run:  cd web ; npm test

import { describe, expect, it } from "vitest";

import { numbersIn, untracedNumbers } from "@/lib/numbers";

describe("numbersIn", () => {
  it("finds whole numbers, decimals and numbers with thousands separators", () => {
    expect(numbersIn("Senna 53% vs 0.92, over 1,000 versions").map((n) => n.value)).toEqual([53, 0.92, 1000]);
  });
  it("remembers how many decimals each number was written with", () => {
    expect(numbersIn("0.9 and 0.88").map((n) => n.decimals)).toEqual([1, 2]);
  });
});

describe("untracedNumbers", () => {
  const tool = { chanceAFinishesAheadPercent: 53, rangePercent: [0.62, 1.13], skillPercent: 0.884 };

  it("accepts numbers quoted from a tool result, rounded or not", () => {
    expect(untracedNumbers("A 53% chance, skill +0.88 (range 0.62 to 1.13), about 0.9", [tool])).toEqual([]);
  });
  it("ignores the sign, so a negative tool value traces a written minus", () => {
    expect(untracedNumbers("an adjustment of −0.12%", [{ wet: -0.12 }])).toEqual([]);
  });
  it("flags a number that no tool returned", () => {
    expect(untracedNumbers("A 61% chance", [tool])).toEqual(["61"]);
  });
  it("accepts numbers that were in the question", () => {
    expect(untracedNumbers("The top 5 are...", [tool], "Who are the top 5?")).toEqual([]);
  });
  it("finds numbers inside text fields of tool results too", () => {
    expect(untracedNumbers("based on 37 listed races", [{ caveat: "37 hand-listed before 2018" }])).toEqual([]);
  });
});

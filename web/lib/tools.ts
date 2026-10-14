// lib/tools.ts
//
// What it does: the tools the Pit Wall assistant can call. Each one looks up
//   or simulates something from the model's exported data and returns plain
//   numbers. The assistant (a language model) is only allowed to quote
//   numbers that came back from these tools, so these functions are where
//   every number in an answer comes from.
//   Numbers are returned ready to quote (percentages as whole numbers, skill
//   to two decimals), so the assistant never has to do arithmetic, and so the
//   checker in lib/numbers.ts can trace every number back to a tool.
// What it reads: the JSON files in public/data (passed in as `ModelData`).
// What it produces: plain JSON-friendly objects.
// Which files use it: app/api/chat/route.ts, eval/run-eval.ts and the tests.

import {
  type Circuit, type Driver, type DriverDetail, type Explanation, headToHead, type HeadToHeads, type Settings,
} from "./data";
import { simulateRace } from "./sim";

// [teammate id, from year, to year, races won, races compared, qualifyings won, qualifyings compared]
export type Records = Record<string, [string, number, number, number, number, number, number][]>;

export type ModelData = {
  drivers: Driver[];
  h2h: HeadToHeads;
  details: Record<string, DriverDetail>;
  records: Records;
  explain: Record<string, Explanation>;
  settings: Settings;
  circuits: Circuit[];
};

const RANGE_LEVEL = 90; // every range is a 90% range
const TYPICAL_LAP_SECONDS = 90;
const MAX_SIMULATIONS = 500;
const MIN_WET_RACES_FOR_LIST = 8;
const NOTE = "Model estimate, not a fact. Cross-era comparisons are especially uncertain.";

/** Lower case, accents removed, only letters: "Häkkinen" -> "hakkinen". */
function plain(text: string): string {
  return text.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z ]/g, "").trim();
}

const round = (value: number, digits = 2) => Number(value.toFixed(digits));

/**
 * Find a driver from a name as a fan would type it: "Senna", "Lewis Hamilton",
 * "Schumacher". When a surname is shared, the driver with more starts wins
 * ("Schumacher" is Michael), unless the first name says otherwise.
 */
export function findDriver(data: ModelData, name: string): Driver | null {
  const wanted = plain(name);
  if (!wanted) return null;
  const exact = data.drivers.find((d) => plain(d.name) === wanted || d.id === wanted.replace(/ /g, "_"));
  if (exact) return exact;
  const words = wanted.split(" ");
  const matches = data.drivers.filter((d) => {
    const full = plain(d.name);
    return words.every((word) => full.split(" ").some((part) => part.startsWith(word)));
  });
  return matches.sort((a, b) => b.starts - a.starts)[0] ?? null;
}

function notFound(name: string) {
  return { error: `No driver called "${name}" is in the data (every F1 World Championship race since 1950).` };
}

function nameOf(data: ModelData, id: string): string {
  return data.drivers.find((d) => d.id === id)?.name ?? id;
}

/** The headline numbers for one driver. */
export function getDriverRating(data: ModelData, args: { name: string }) {
  const driver = findDriver(data, args.name);
  if (!driver) return notFound(args.name);
  return {
    driver: driver.name,
    peakSkillPercent: round(driver.skill),
    rangePercent: [round(driver.skill - 1.645 * driver.sd), round(driver.skill + 1.645 * driver.sd)],
    rangeLevel: RANGE_LEVEL,
    skillMeaning: "percent of lap time quicker than an average newcomer of today, in the same car",
    peakSeasons: driver.peak,
    rank: driver.rank,
    rankRange: driver.rank === null ? null : [driver.rankLow, driver.rankHigh],
    rankedAmong: "drivers with at least 30 starts",
    careerSeasons: driver.years,
    starts: driver.starts,
    wins: driver.wins,
    consistency: driver.consistency,
    consistencyMeaning: "0 to 100, 50 is typical",
    note: NOTE,
  };
}

/** The chance A beats B in identical cars, each at their peak, with its range. */
export function compareDrivers(data: ModelData, args: { driver_a: string; driver_b: string }) {
  const a = findDriver(data, args.driver_a);
  const b = findDriver(data, args.driver_b);
  if (!a) return notFound(args.driver_a);
  if (!b) return notFound(args.driver_b);
  if (a.id === b.id) return { error: "Those are the same driver." };
  const gap = a.skill - b.skill;
  const stored = headToHead(data.h2h, a.id, b.id);
  let chance = stored;
  let method = "precomputed from 1,000 plausible versions of both ratings";
  if (!chance) {
    // One of them has fewer than 30 starts: approximate, treating the two
    // uncertainties as unrelated (which makes the range somewhat too wide).
    const slope = data.settings.raceSlope;
    const sd = Math.sqrt(a.sd ** 2 + b.sd ** 2);
    const curve = (g: number) => 100 / (1 + Math.exp(-slope * g));
    const softened = (slope * gap) / Math.sqrt(1 + (Math.PI * slope ** 2 * sd ** 2) / 8);
    chance = { best: Math.round(100 / (1 + Math.exp(-softened))), low: Math.round(curve(gap - 1.645 * sd)), high: Math.round(curve(gap + 1.645 * sd)) };
    method = "approximate (one driver has fewer than 30 starts, so no precomputed value)";
  }
  const record = data.records[a.id]?.find((row) => row[0] === b.id);
  return {
    driverA: a.name,
    driverB: b.name,
    chanceAFinishesAheadPercent: chance.best,
    chanceBFinishesAheadPercent: 100 - chance.best,
    chanceARangePercent: [chance.low, chance.high],
    rangeLevel: RANGE_LEVEL,
    skillAPercent: round(a.skill),
    skillBPercent: round(b.skill),
    skillGapPercent: round(gap),
    skillGapSecondsPer90sLap: round((gap / 100) * TYPICAL_LAP_SECONDS),
    peakSeasonsA: a.peak,
    peakSeasonsB: b.peak,
    wereTeammates: Boolean(record),
    method,
    note: NOTE,
  };
}

/** Run many equal car races in the browser simulator and summarise them. */
export function simulateRaces(data: ModelData, args: { drivers: string[]; circuit?: string; races?: number }) {
  const found = args.drivers.map((name) => ({ name, driver: findDriver(data, name) }));
  const missing = found.filter((entry) => !entry.driver).map((entry) => entry.name);
  if (missing.length) return { error: `Not in the data: ${missing.join(", ")}.` };
  const drivers = [...new Map(found.map((entry) => [entry.driver!.id, entry.driver!])).values()];
  if (drivers.length < 2 || drivers.length > 20) return { error: "A race needs between 2 and 20 different drivers." };
  const wanted = plain(args.circuit ?? "silverstone");
  const circuit = data.circuits.find((c) => plain(`${c.name} ${c.country} ${c.id}`).includes(wanted)) ?? null;
  if (!circuit) return { error: `No circuit matching "${args.circuit}". Circuits raced since 2018 are available.` };
  const races = Math.min(Math.max(args.races ?? MAX_SIMULATIONS, 50), MAX_SIMULATIONS);
  const difficulty = data.settings.difficulty[circuit.id] ?? 0.5;
  const wins = new Array(drivers.length).fill(0);
  const podiums = new Array(drivers.length).fill(0);
  const places = new Array(drivers.length).fill(0);
  for (let i = 0; i < races; i++) {
    const race = simulateRace(drivers, circuit.lapSeconds, circuit.raceLaps, difficulty, data.settings.sim);
    race.finishOrder.forEach((car, place) => {
      places[car] += place + 1;
      if (place === 0) wins[car] += 1;
      if (place < 3) podiums[car] += 1;
    });
  }
  return {
    circuit: circuit.name,
    laps: circuit.raceLaps,
    overtakingDifficulty: round(difficulty),
    overtakingDifficultyMeaning: "0 easiest, 1 hardest (Monaco)",
    simulatedRaces: races,
    grid: "simulated qualifying",
    results: drivers.map((driver, car) => ({
      driver: driver.name,
      winPercent: Math.round((100 * wins[car]) / races),
      podiumPercent: Math.round((100 * podiums[car]) / races),
      averageFinish: round(places[car] / races, 1),
    })).sort((x, y) => y.winPercent - x.winPercent),
    note: "Simulated from the model with identical cars. Not a prediction of a real race. " + NOTE,
  };
}

/** Two drivers' real record as teammates. */
export function getTeammateRecord(data: ModelData, args: { driver_a: string; driver_b: string }) {
  const a = findDriver(data, args.driver_a);
  const b = findDriver(data, args.driver_b);
  if (!a) return notFound(args.driver_a);
  if (!b) return notFound(args.driver_b);
  const row = data.records[a.id]?.find((r) => r[0] === b.id);
  if (!row) return { driverA: a.name, driverB: b.name, wereTeammates: false };
  const [, from, to, raceWins, races, qualiWins, qualis] = row;
  return {
    driverA: a.name, driverB: b.name, wereTeammates: true, seasons: [from, to],
    raceFinishedAheadA: raceWins, racesCompared: races, raceFinishedAheadB: races - raceWins,
    qualifiedAheadA: qualiWins, qualifyingsCompared: qualis, qualifiedAheadB: qualis - qualiWins,
    countingRule: "races where either car broke down are not counted",
    note: "These are real results, not model estimates.",
  };
}

/** Why a driver has their rating: the battles that moved it most, and the chain to today's grid. */
export function explainRating(data: ModelData, args: { name: string }) {
  const driver = findDriver(data, args.name);
  if (!driver) return notFound(args.name);
  const entry = data.explain[driver.id];
  if (!entry) return { driver: driver.name, error: "No explanation available for this driver." };
  const battle = ([a, b, from, to, races, effect]: Explanation["up"][number]) => ({
    battle: `${nameOf(data, a)} vs ${nameOf(data, b)}`, seasons: [from, to], races,
    effectOnRatingPercent: round(effect, 3),
  });
  return {
    driver: driver.name,
    peakSkillPercent: round(driver.skill),
    rank: driver.rank,
    rankRange: driver.rank === null ? null : [driver.rankLow, driver.rankHigh],
    pushedRatingUp: entry.up.map(battle),
    pulledRatingDown: entry.down.map(battle),
    chainToTodaysGrid: entry.chain.map(([next, from, to, races]) => ({ teammateOf: nameOf(data, next), seasons: [from, to], races })),
    method: "effect = rating with that pairing's results minus rating without them (an approximation)",
    note: NOTE,
  };
}

/** A driver's wet weather adjustment, or the best wet weather drivers. */
export function getWetRating(data: ModelData, args: { name?: string }) {
  const caveat = "Based on few wet races: 37 hand-listed before 2018 and 17 measured since. Very uncertain.";
  if (args.name) {
    const driver = findDriver(data, args.name);
    if (!driver) return notFound(args.name);
    const [adjustment, sd, races] = driver.wet;
    return {
      driver: driver.name, wetAdjustmentPercent: round(adjustment),
      rangePercent: [round(adjustment - 1.645 * sd), round(adjustment + 1.645 * sd)], rangeLevel: RANGE_LEVEL,
      wetRacesAgainstTeammates: races,
      meaning: "added to the driver's skill in wet races; positive = better in the wet than in the dry",
      caveat, note: NOTE,
    };
  }
  const listed = data.drivers.filter((d) => d.wet[2] >= MIN_WET_RACES_FOR_LIST)
    .sort((a, b) => b.skill + b.wet[0] - (a.skill + a.wet[0])).slice(0, 10);
  return {
    bestInTheWet: listed.map((d) => ({
      driver: d.name, wetSkillPercent: round(d.skill + d.wet[0]), drySkillPercent: round(d.skill),
      wetAdjustmentPercent: round(d.wet[0]), wetRacesAgainstTeammates: d.wet[2],
    })),
    rankedBy: "peak skill plus wet adjustment",
    onlyDriversWith: `at least ${MIN_WET_RACES_FOR_LIST} wet races against a teammate`,
    caveat, note: NOTE,
  };
}

/** The top drivers overall or in one decade. */
export function listTopDrivers(data: ModelData, args: { decade?: number; count?: number }) {
  const count = Math.min(Math.max(args.count ?? 10, 1), 20);
  const pool = data.drivers.filter((d) => d.rank !== null
    && (args.decade === undefined || Math.floor(d.peak[0] / 10) * 10 === args.decade));
  return {
    decade: args.decade ?? "all time",
    drivers: pool.slice(0, count).map((d) => ({
      driver: d.name, peakSkillPercent: round(d.skill), rank: d.rank, rankRange: [d.rankLow, d.rankHigh], peakSeasons: d.peak,
    })),
    rankedAmong: "drivers with at least 30 starts", note: NOTE,
  };
}

// The descriptions are what the language model reads to decide which tool to call.
export const TOOL_DEFINITIONS = [
  {
    name: "get_driver_rating",
    description: "A driver's model rating: peak skill with its 90% range, rank and plausible rank range, peak seasons, starts, wins and consistency. Use for any question about how good one driver is.",
    input_schema: { type: "object", properties: { name: { type: "string", description: "Driver name, e.g. 'Ayrton Senna' or 'Senna'" } }, required: ["name"], additionalProperties: false },
  },
  {
    name: "compare_drivers",
    description: "The chance driver A finishes ahead of driver B in identical cars, each at their peak, with a 90% range and their skill gap. Use for 'would X beat Y', 'who is better, X or Y'.",
    input_schema: { type: "object", properties: { driver_a: { type: "string" }, driver_b: { type: "string" } }, required: ["driver_a", "driver_b"], additionalProperties: false },
  },
  {
    name: "simulate_race",
    description: "Simulate up to 500 equal car races between 2 to 20 drivers at a circuit (full race distance, simulated qualifying) and return win %, podium % and average finish for each. Use for questions about a specific circuit or a group of drivers.",
    input_schema: {
      type: "object",
      properties: {
        drivers: { type: "array", items: { type: "string" }, description: "2 to 20 driver names" },
        circuit: { type: "string", description: "Circuit or country, e.g. 'Monaco', 'Monza', 'Interlagos'. Circuits raced since 2018 only." },
        races: { type: "integer", description: "How many races to simulate, 50 to 500 (default 500)" },
      },
      required: ["drivers", "circuit"], additionalProperties: false,
    },
  },
  {
    name: "get_teammate_record",
    description: "The real head to head record of two drivers who were teammates: races and qualifying sessions where each was ahead. Real results, not estimates.",
    input_schema: { type: "object", properties: { driver_a: { type: "string" }, driver_b: { type: "string" } }, required: ["driver_a", "driver_b"], additionalProperties: false },
  },
  {
    name: "explain_rating",
    description: "Why a driver has their rating: the teammate battles that pushed it up and pulled it down most, and the chain of teammates linking them to today's grid. Use for 'why is X ranked where they are'.",
    input_schema: { type: "object", properties: { name: { type: "string" } }, required: ["name"], additionalProperties: false },
  },
  {
    name: "get_wet_rating",
    description: "A driver's wet weather skill adjustment with its range and how many wet races it rests on. Without a name, lists the best wet weather drivers.",
    input_schema: { type: "object", properties: { name: { type: "string", description: "Leave out to get the list of best wet weather drivers" } }, additionalProperties: false },
  },
  {
    name: "list_top_drivers",
    description: "The top rated drivers of all time, or of one decade (by when their peak began), with ranges.",
    input_schema: {
      type: "object",
      properties: { decade: { type: "integer", description: "e.g. 1980 for the 1980s; leave out for all time" }, count: { type: "integer", description: "1 to 20, default 10" } },
      additionalProperties: false,
    },
  },
] as const;

export type ToolName = (typeof TOOL_DEFINITIONS)[number]["name"];

/** Run one tool by name. Unknown tools and bad input come back as an error object, never a crash. */
export function runTool(data: ModelData, name: string, input: unknown): Record<string, unknown> {
  const args = (input ?? {}) as Record<string, never>;
  try {
    switch (name) {
      case "get_driver_rating": return getDriverRating(data, args);
      case "compare_drivers": return compareDrivers(data, args);
      case "simulate_race": return simulateRaces(data, args);
      case "get_teammate_record": return getTeammateRecord(data, args);
      case "explain_rating": return explainRating(data, args);
      case "get_wet_rating": return getWetRating(data, args);
      case "list_top_drivers": return listTopDrivers(data, args);
      default: return { error: `Unknown tool ${name}` };
    }
  } catch (error) {
    return { error: `The tool failed: ${(error as Error).message}` };
  }
}

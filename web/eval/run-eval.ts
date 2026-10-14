// eval/run-eval.ts
//
// What it does: scores the Pit Wall assistant on the 50 questions in
//   eval/questions.json. For each answer it checks four things:
//     1. tool:       did it call one of the tools a correct answer needs?
//     2. numbers:    does the answer quote the numbers it should (the tool
//                    result fields listed for that question)?
//     3. invented:   is every number in the answer traceable to a tool result
//                    or the question? (lib/numbers.ts)
//     4. refusal:    for the 5 trick questions, did it decline, without
//                    inventing numbers?
//   It prints a table and writes eval/results.json. Each run costs money
//   (about 50 questions x a few model calls), so it only runs when asked.
// What it reads: eval/questions.json, the model data, ANTHROPIC_API_KEY.
// What it produces: eval/results.json and a printed summary.
// Which files use it: none. The README quotes its score.
//
// Run:  cd web ; npm run eval            (needs ANTHROPIC_API_KEY)
//       cd web ; npm run eval -- --dry   (free: checks the questions and the tools only)

import fs from "node:fs";

import Anthropic from "@anthropic-ai/sdk";

import { MODEL_DATA } from "../lib/modelData";
import { numbersIn, untracedNumbers } from "../lib/numbers";
import { askPitWall, DEFAULT_MODEL, type ToolCall } from "../lib/pitwall";
import { runTool } from "../lib/tools";

type Expect = { tool: string; input?: Record<string, unknown>; fields: string[] };
type Question = {
  id: string; category: "factual" | "comparison" | "trick"; question: string;
  tools: string[]; expect: Expect[]; known?: Record<string, number>;
};

const QUESTIONS: Question[] = JSON.parse(fs.readFileSync("eval/questions.json", "utf-8")).questions;
const DECLINE_WORDS = /not (in|found|available|able)|no driver|can'?t|cannot|don'?t have|doesn'?t have|unable|no data|isn'?t|does not|do not|unknown|future|hasn'?t happened|not yet/i;

/** Every value of a field anywhere inside a tool result (results can be nested lists). */
function fieldValues(value: unknown, field: string, out: number[] = []): number[] {
  if (Array.isArray(value)) value.forEach((item) => fieldValues(item, field, out));
  else if (value && typeof value === "object") {
    for (const [key, inner] of Object.entries(value)) {
      if (key === field && typeof inner === "number") out.push(Math.abs(inner));
      else fieldValues(inner, field, out);
    }
  }
  return out;
}

/** True if the answer mentions this number at some precision ("0.92" for 0.921, "53" for 53). */
function mentions(answer: string, value: number): boolean {
  return numbersIn(answer).some((n) => Math.abs(n.value - value) <= 0.5 * 10 ** -n.decimals + 1e-9);
}

/** The tool result an expected number should come from: the assistant's own call, or a fresh one. */
function resultFor(expect: Expect, calls: ToolCall[]): Record<string, unknown> {
  const own = calls.filter((call) => call.name === expect.tool && !("error" in call.result));
  if (expect.tool === "simulate_race") return own[0]?.result ?? {}; // random: only its own run counts
  return expect.input ? runTool(MODEL_DATA, expect.tool, expect.input) : own[0]?.result ?? {};
}

function score(q: Question, answer: string, calls: ToolCall[]) {
  const used = calls.map((call) => call.name);
  const untraced = untracedNumbers(answer, calls.map((call) => call.result), q.question);
  const checks: Record<string, boolean | null> = {
    tool: q.tools.length === 0 ? null : q.tools.some((tool) => used.includes(tool)),
    numbers: q.expect.length === 0 ? null : q.expect.every((e) => {
      const result = resultFor(e, calls);
      // For a field with several values (a list of drivers), quoting any one counts.
      return e.fields.every((field) => fieldValues(result, field).some((value) => mentions(answer, value)));
    }),
    invented: untraced.length === 0,
    refusal: q.category === "trick" ? DECLINE_WORDS.test(answer) && untraced.length === 0 : null,
  };
  return { checks, untraced, used };
}

/** Free checks: every expected tool call works, and the known real-world facts match the data. */
function dryRun(): boolean {
  let ok = true;
  for (const q of QUESTIONS) {
    for (const e of q.expect.filter((x) => x.input)) {
      const result = runTool(MODEL_DATA, e.tool, e.input);
      for (const field of e.fields) {
        if (fieldValues(result, field).length === 0) {
          console.log(`${q.id}: ${e.tool} ${JSON.stringify(e.input)} has no "${field}": ${JSON.stringify(result).slice(0, 160)}`);
          ok = false;
        }
      }
      for (const [field, value] of Object.entries(q.known ?? {})) {
        if (!fieldValues(result, field).includes(value)) {
          console.log(`${q.id}: expected ${field} = ${value}, the data says ${fieldValues(result, field)}`);
          ok = false;
        }
      }
    }
  }
  const counts = QUESTIONS.reduce<Record<string, number>>((all, q) => ({ ...all, [q.category]: (all[q.category] ?? 0) + 1 }), {});
  console.log(`${QUESTIONS.length} questions: ${JSON.stringify(counts)}. Dry run ${ok ? "passed" : "FAILED"}.`);
  return ok;
}

async function main() {
  if (process.argv.includes("--dry")) process.exit(dryRun() ? 0 : 1);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.log("ANTHROPIC_API_KEY is not set, so the assistant cannot be evaluated. Try --dry for the free checks.");
    process.exit(1);
  }
  const client = new Anthropic();
  const model = process.env.PIT_WALL_MODEL || DEFAULT_MODEL;
  const rows = [];
  for (const q of QUESTIONS) {
    const result = await askPitWall(client, MODEL_DATA, q.question, [], model);
    const { checks, untraced, used } = score(q, result.answer, result.toolCalls);
    rows.push({ id: q.id, category: q.category, question: q.question, answer: result.answer, used, untraced, rewritten: result.rewritten, checks });
    const marks = Object.entries(checks).filter(([, v]) => v !== null).map(([k, v]) => `${k}:${v ? "ok" : "FAIL"}`).join(" ");
    console.log(`${q.id} ${marks}`);
  }
  const rate = (name: string) => {
    const scored = rows.filter((row) => row.checks[name] !== null);
    return { passed: scored.filter((row) => row.checks[name]).length, of: scored.length };
  };
  const summary = {
    model, questions: rows.length,
    tool: rate("tool"), numbers: rate("numbers"), invented: rate("invented"), refusal: rate("refusal"),
    allChecksPassed: rows.filter((row) => Object.values(row.checks).every((v) => v !== false)).length,
    rewrittenOnce: rows.filter((row) => row.rewritten).length,
  };
  fs.writeFileSync("eval/results.json", JSON.stringify({ summary, rows }, null, 1));
  console.log("\n", summary);
}

main();

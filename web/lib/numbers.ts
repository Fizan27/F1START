// lib/numbers.ts
//
// What it does: checks that every number in an assistant answer came from a
//   tool result (or from the question). This is the safety net behind the
//   rule "the assistant must not invent numbers": the chat route uses it to
//   catch an answer with an untraceable number and ask for a rewrite, and the
//   evaluation script uses it to score answers.
//   A number in the answer counts as traced when some tool number rounds to
//   it at the precision the answer used: a tool value of 0.884 traces "0.88"
//   and "0.9"; a tool value of 53 traces "53%".
// What it reads: nothing.
// What it produces: lists of numbers.
// Which files use it: app/api/chat/route.ts, eval/run-eval.ts and the tests.

/** Every number written in a piece of text, with how many decimals it was written with. */
export function numbersIn(text: string): { value: number; decimals: number; raw: string }[] {
  const found = [];
  // Digits with an optional decimal part. Thousands separators ("1,000") are joined first.
  const cleaned = text.replace(/(\d),(\d{3})\b/g, "$1$2");
  for (const match of cleaned.matchAll(/\d+(?:\.\d+)?/g)) {
    const raw = match[0];
    found.push({ value: Number(raw), decimals: raw.includes(".") ? raw.split(".")[1].length : 0, raw });
  }
  return found;
}

/** Every number anywhere inside a tool result (or any JSON value). */
export function numbersInValue(value: unknown, out: number[] = []): number[] {
  if (typeof value === "number") out.push(Math.abs(value));
  else if (typeof value === "string") numbersIn(value).forEach((n) => out.push(n.value));
  else if (Array.isArray(value)) value.forEach((item) => numbersInValue(item, out));
  else if (value && typeof value === "object") Object.values(value).forEach((item) => numbersInValue(item, out));
  return out;
}

/**
 * Numbers in `answer` that cannot be traced to `sources` (tool results and
 * the question). Signs are ignored: "−0.12%" is traced by -0.12.
 */
export function untracedNumbers(answer: string, sources: unknown[], question = ""): string[] {
  const known = sources.flatMap((source) => numbersInValue(source));
  numbersIn(question).forEach((n) => known.push(n.value));
  return numbersIn(answer)
    .filter(({ value, decimals }) => {
      const tolerance = 0.5 * 10 ** -decimals + 1e-9;
      return !known.some((source) => Math.abs(source - value) <= tolerance);
    })
    .map((n) => n.raw);
}

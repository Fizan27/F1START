// lib/pitwall.ts
//
// What it does: the Pit Wall assistant's brain. It sends a fan's question to
//   Claude (Anthropic's language model) together with the tools in
//   lib/tools.ts, runs whatever tools Claude asks for, and hands the results
//   back until Claude writes its answer.
//   Then it checks the answer: every number in it must appear in a tool
//   result or in the question (lib/numbers.ts). If one does not, Claude is
//   told exactly which numbers could not be traced and asked to rewrite the
//   answer once. If the rewrite still has untraceable numbers, the answer is
//   returned with those numbers listed, so the page can warn the reader.
// What it reads: nothing from disk. The caller passes the API client and data.
// What it produces: the answer, every tool call made, and the check result.
// Which files use it: app/api/chat/route.ts and eval/run-eval.ts.

import type Anthropic from "@anthropic-ai/sdk";

import { untracedNumbers } from "./numbers";
import { type ModelData, runTool, TOOL_DEFINITIONS } from "./tools";

export const DEFAULT_MODEL = "claude-opus-5-5";
const MAX_ROUNDS = 6; // model calls per question, including tool rounds and the one rewrite
const MAX_TOKENS = 2000;

export const SYSTEM_PROMPT = `You are Pit Wall, the assistant on F1 Equalizer, a site that rates every Formula 1 driver since 1950 as if they all drove the same car. The ratings come from a statistical model built on teammate comparisons.

How to answer:
- Get every number from the tools. Never state a number (a percentage, skill, rank, count, year or time) that is not in a tool result or in the question. Do not do arithmetic on tool numbers; quote them as they are. If you would need a number no tool gives, say the site does not have it.
- Always give the uncertainty: quote the range the tool returns, and say these are model estimates, not facts. Comparisons across eras are especially uncertain.
- If a driver is not found, or the question is about something the model cannot know (a future season, a made-up driver, a real race that has not happened, anything that is not about F1 drivers), say so plainly and do not guess.
- Answer in plain English in at most about 120 words. No tables, no markdown headings.
- Skill is in percent of lap time quicker than an average newcomer of today in the same car.`;

export type ToolCall = { name: string; input: unknown; result: Record<string, unknown> };
export type PitWallAnswer = {
  answer: string;
  toolCalls: ToolCall[];
  untraced: string[]; // numbers in the final answer that no tool result contains
  rewritten: boolean; // true if the answer had to be rewritten once
  stopReason: string | null;
};
export type Turn = { question: string; answer: string };

function textOf(content: Anthropic.Beta.Messages.BetaContentBlock[]): string {
  return content.filter((block) => block.type === "text").map((block) => (block as { text: string }).text).join("\n").trim();
}

/** Ask one question. `history` holds earlier questions and answers of the same chat (text only). */
export async function askPitWall(
  client: Anthropic, data: ModelData, question: string, history: Turn[] = [], model = DEFAULT_MODEL,
): Promise<PitWallAnswer> {
  const messages: Anthropic.Beta.Messages.BetaMessageParam[] = [];
  for (const turn of history) {
    messages.push({ role: "user", content: turn.question }, { role: "assistant", content: turn.answer });
  }
  messages.push({ role: "user", content: question });

  const toolCalls: ToolCall[] = [];
  let rewritten = false;
  let answer = "";
  let stopReason: string | null = null;

  for (let round = 0; round < MAX_ROUNDS; round++) {
    const response = await client.beta.messages.create({
      model,
      max_tokens: MAX_TOKENS,
      // The system prompt and tools never change, so they are cached between
      // questions, which makes every question after the first cheaper.
      system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
      tools: TOOL_DEFINITIONS as unknown as Anthropic.Beta.Messages.BetaTool[],
      messages,
      output_config: { effort: "low" }, // looking up numbers needs little deliberation
      // If a safety check declines the request, Anthropic re-runs it on a
      // suitable fallback model instead of returning a refusal.
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
    });
    stopReason = response.stop_reason;
    if (response.stop_reason === "refusal") {
      return { answer: "Sorry, I can't help with that one.", toolCalls, untraced: [], rewritten, stopReason };
    }
    messages.push({ role: "assistant", content: response.content });

    const uses = response.content.filter((block) => block.type === "tool_use") as Anthropic.Beta.Messages.BetaToolUseBlock[];
    if (uses.length > 0 && response.stop_reason === "tool_use") {
      // All results go back in ONE user message, as the API expects.
      const results = uses.map((use) => {
        const result = runTool(data, use.name, use.input);
        toolCalls.push({ name: use.name, input: use.input, result });
        return { type: "tool_result" as const, tool_use_id: use.id, content: JSON.stringify(result), is_error: "error" in result };
      });
      messages.push({ role: "user", content: results });
      continue;
    }

    answer = textOf(response.content);
    const untraced = untracedNumbers(answer, toolCalls.map((call) => call.result), question);
    if (untraced.length === 0 || rewritten) {
      return { answer, toolCalls, untraced, rewritten, stopReason };
    }
    // One chance to fix it: say exactly which numbers had no source.
    rewritten = true;
    messages.push({
      role: "user",
      content: `Check: these numbers in your answer are not in any tool result: ${untraced.join(", ")}. `
        + "Rewrite the answer using only numbers that appear in tool results (call a tool if you need one), or leave them out.",
    });
  }
  return {
    answer: answer || "Sorry, I couldn't finish that answer. Please try a simpler question.",
    toolCalls, untraced: untracedNumbers(answer, toolCalls.map((call) => call.result), question), rewritten, stopReason,
  };
}

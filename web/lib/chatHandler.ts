// lib/chatHandler.ts
//
// What it does: handles one chat request from the browser: checks the
//   assistant is switched on (an API key is set), checks the question is
//   sensible and the visitor is within the rate limits, then asks the
//   assistant. Kept separate from the Next.js route so the tests can call it
//   with a fake API client and never spend money.
// What it reads: the request body ({ question, history }).
// What it produces: { status, body } for the route to send back.
// Which files use it: app/api/chat/route.ts and the tests.

import Anthropic from "@anthropic-ai/sdk";

import { askPitWall, type Turn } from "./pitwall";
import type { RateLimiter } from "./rateLimit";
import type { ModelData } from "./tools";

const MAX_QUESTION_LENGTH = 500;
const MAX_HISTORY = 4;
const MAX_HISTORY_TEXT = 1500;

export type ChatDependencies = {
  apiKey: string | undefined;
  makeClient: (apiKey: string) => Anthropic;
  data: ModelData;
  limiter: RateLimiter;
  model?: string;
};

export type ChatResult = { status: number; body: Record<string, unknown> };

/** Keep only well-formed earlier turns, the last few, trimmed to a sensible length. */
function cleanHistory(history: unknown): Turn[] {
  if (!Array.isArray(history)) return [];
  return history
    .filter((turn) => turn && typeof turn.question === "string" && typeof turn.answer === "string")
    .slice(-MAX_HISTORY)
    .map((turn) => ({ question: turn.question.slice(0, MAX_QUESTION_LENGTH), answer: turn.answer.slice(0, MAX_HISTORY_TEXT) }));
}

export async function handleChat(body: unknown, visitor: string, deps: ChatDependencies): Promise<ChatResult> {
  if (!deps.apiKey) {
    return { status: 503, body: { error: "The Pit Wall assistant is not available on this copy of the site." } };
  }
  const { question, history } = (body ?? {}) as { question?: unknown; history?: unknown };
  if (typeof question !== "string" || !question.trim()) {
    return { status: 400, body: { error: "Please type a question." } };
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return { status: 400, body: { error: `Please keep questions under ${MAX_QUESTION_LENGTH} characters.` } };
  }
  const refusal = deps.limiter.check(visitor);
  if (refusal) return { status: 429, body: { error: refusal } };

  try {
    const result = await askPitWall(deps.makeClient(deps.apiKey), deps.data, question.trim(), cleanHistory(history), deps.model);
    return {
      status: 200,
      body: {
        answer: result.answer,
        tools: result.toolCalls.map((call) => ({ name: call.name, input: call.input })),
        untraced: result.untraced,
      },
    };
  } catch (error) {
    if (error instanceof Anthropic.RateLimitError) {
      return { status: 503, body: { error: "The assistant is busy right now. Please try again in a minute." } };
    }
    if (error instanceof Anthropic.AuthenticationError) {
      return { status: 503, body: { error: "The Pit Wall assistant is not available on this copy of the site." } };
    }
    if (error instanceof Anthropic.APIError) {
      return { status: 502, body: { error: "The assistant could not answer just now. Please try again." } };
    }
    throw error;
  }
}

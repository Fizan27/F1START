// app/api/chat/route.ts
//
// What it does: the server endpoint behind the Pit Wall chat panel. It runs
//   on Vercel as a small serverless function, so the Anthropic API key stays
//   on the server and never reaches the browser.
//   GET  /api/chat  -> { available: true/false } (is an API key set?)
//   POST /api/chat  -> { answer, tools, untraced } or { error }
// What it reads: the ANTHROPIC_API_KEY environment variable (set in Vercel's
//   project settings, never in the code), optionally PIT_WALL_MODEL and the
//   rate limit variables in lib/rateLimit.ts.
// What it produces: JSON responses.
// Which files use it: components/PitWall.tsx calls it.

import Anthropic from "@anthropic-ai/sdk";
import { NextResponse } from "next/server";

import { handleChat } from "@/lib/chatHandler";
import { MODEL_DATA } from "@/lib/modelData";
import { DEFAULT_MODEL } from "@/lib/pitwall";
import { limitsFromEnvironment, RateLimiter } from "@/lib/rateLimit";

export const runtime = "nodejs";
export const maxDuration = 60; // seconds: a question with several tool calls can take a while

// One limiter per running copy of the server (see lib/rateLimit.ts).
const limiter = new RateLimiter(limitsFromEnvironment());

export async function GET() {
  return NextResponse.json({ available: Boolean(process.env.ANTHROPIC_API_KEY) });
}

export async function POST(request: Request) {
  // Vercel puts the visitor's address first in x-forwarded-for.
  const visitor = request.headers.get("x-forwarded-for")?.split(",")[0].trim() || "unknown";
  const body = await request.json().catch(() => null);
  const result = await handleChat(body, visitor, {
    apiKey: process.env.ANTHROPIC_API_KEY,
    makeClient: (apiKey) => new Anthropic({ apiKey }),
    data: MODEL_DATA,
    limiter,
    model: process.env.PIT_WALL_MODEL || DEFAULT_MODEL,
  });
  return NextResponse.json(result.body, { status: result.status });
}

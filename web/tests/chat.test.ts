// tests/chat.test.ts
//
// What it does: checks the chat request handler and the assistant loop with a
//   FAKE API client that replies from a script, so no money is ever spent:
//   no key -> unavailable, bad input -> refused, tool calls are run and fed
//   back, and an answer with an invented number is sent back for a rewrite.
// Which files use it: none.  Run:  cd web ; npm test

import type Anthropic from "@anthropic-ai/sdk";
import { describe, expect, it } from "vitest";

import { handleChat } from "@/lib/chatHandler";
import { MODEL_DATA } from "@/lib/modelData";
import { RateLimiter } from "@/lib/rateLimit";

type Reply = { stop_reason: string; content: Record<string, unknown>[] };

/** A stand-in for the Anthropic client: returns the scripted replies in order and records requests. */
function fakeClient(replies: Reply[]) {
  const requests: Record<string, unknown>[] = [];
  const client = {
    beta: { messages: { create: async (request: Record<string, unknown>) => {
      requests.push(JSON.parse(JSON.stringify(request)));
      return replies.shift();
    } } },
  };
  return { client: client as unknown as Anthropic, requests };
}

const generous = () => new RateLimiter({ perTenMinutes: 100, perDay: 100, sitePerDay: 100 });
const senna = MODEL_DATA.drivers.find((d) => d.id === "senna")!;
const toolCall: Reply = {
  stop_reason: "tool_use",
  content: [{ type: "tool_use", id: "call_1", name: "get_driver_rating", input: { name: "Senna" } }],
};

function deps(client: Anthropic, apiKey: string | undefined = "test-key") {
  return { apiKey, makeClient: () => client, data: MODEL_DATA, limiter: generous() };
}

describe("handleChat", () => {
  it("says the assistant is unavailable when no API key is set, without calling the API", async () => {
    const { client, requests } = fakeClient([]);
    const result = await handleChat({ question: "Hi" }, "v", deps(client, undefined));
    expect(result.status).toBe(503);
    expect(String(result.body.error)).toMatch(/not available/);
    expect(requests).toHaveLength(0);
  });

  it("refuses empty and over-long questions", async () => {
    const { client } = fakeClient([]);
    expect((await handleChat({ question: "  " }, "v", deps(client))).status).toBe(400);
    expect((await handleChat({ question: "x".repeat(501) }, "v", deps(client))).status).toBe(400);
    expect((await handleChat(null, "v", deps(client))).status).toBe(400);
  });

  it("enforces the rate limit", async () => {
    const { client } = fakeClient([{ stop_reason: "end_turn", content: [{ type: "text", text: "Hello." }] }]);
    const limiter = new RateLimiter({ perTenMinutes: 1, perDay: 10, sitePerDay: 10 });
    const options = { ...deps(client), limiter };
    expect((await handleChat({ question: "One" }, "v", options)).status).toBe(200);
    expect((await handleChat({ question: "Two" }, "v", options)).status).toBe(429);
  });

  it("runs the tool the model asks for and returns an answer built from it", async () => {
    const wins = String(senna.wins);
    const { client, requests } = fakeClient([
      toolCall,
      { stop_reason: "end_turn", content: [{ type: "text", text: `Senna won ${wins} races. This is a model estimate.` }] },
    ]);
    const result = await handleChat({ question: "How many races did Senna win?" }, "v", deps(client));
    expect(result.status).toBe(200);
    expect(result.body.untraced).toEqual([]);
    expect(result.body.tools).toEqual([{ name: "get_driver_rating", input: { name: "Senna" } }]);
    // The tool result went back to the model in the second request.
    const second = requests[1].messages as { role: string; content: unknown }[];
    expect(JSON.stringify(second.at(-1))).toContain('"wins":41');
    // Safety settings are on every request.
    expect(requests[0].fallbacks).toBe("default");
  });

  it("sends an answer with an invented number back for one rewrite", async () => {
    const { client, requests } = fakeClient([
      toolCall,
      { stop_reason: "end_turn", content: [{ type: "text", text: "Senna would win 77% of races." }] },
      { stop_reason: "end_turn", content: [{ type: "text", text: `Senna won ${senna.wins} races.` }] },
    ]);
    const result = await handleChat({ question: "How good was Senna?" }, "v", deps(client));
    expect(result.body.untraced).toEqual([]);
    expect(result.body.answer).toContain(String(senna.wins));
    expect(JSON.stringify(requests[2].messages)).toContain("77");
  });

  it("reports numbers it still cannot trace after the rewrite", async () => {
    const { client } = fakeClient([
      toolCall,
      { stop_reason: "end_turn", content: [{ type: "text", text: "Senna would win 77% of races." }] },
      { stop_reason: "end_turn", content: [{ type: "text", text: "Senna would win 78% of races." }] },
    ]);
    const result = await handleChat({ question: "How good was Senna?" }, "v", deps(client));
    expect(result.body.untraced).toEqual(["78"]);
  });
});

// components/PitWall.tsx
//
// What it does: the Pit Wall chat panel. A button in the corner of every page
//   opens it; fans type plain English questions and get answers built only
//   from the model's numbers (see lib/pitwall.ts). If the site has no API key,
//   the panel says the assistant is unavailable instead of failing.
// What it reads: GET and POST /api/chat.
// Which files use it: app/layout.tsx.

"use client";

import { useEffect, useRef, useState } from "react";

type Message = { question: string; answer?: string; error?: string; tools?: string[]; untraced?: string[] };

const EXAMPLES = [
  "Would Senna beat Verstappen at Monaco in equal cars?",
  "Why is Hamilton ranked where he is?",
  "Who is the best wet weather driver?",
];

const TOOL_LABELS: Record<string, string> = {
  get_driver_rating: "rating", compare_drivers: "head to head", simulate_race: "race simulation",
  get_teammate_record: "teammate record", explain_rating: "rating explanation",
  get_wet_rating: "wet rating", list_top_drivers: "ranking",
};

export default function PitWall() {
  const [open, setOpen] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || available !== null) return;
    fetch("/api/chat").then((r) => r.json()).then((d) => setAvailable(Boolean(d.available))).catch(() => setAvailable(false));
  }, [open, available]);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  async function ask(text: string) {
    const asked = text.trim();
    if (!asked || busy) return;
    setQuestion("");
    setBusy(true);
    const history = messages.filter((m) => m.answer).map((m) => ({ question: m.question, answer: m.answer }));
    setMessages((list) => [...list, { question: asked }]);
    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: asked, history }),
      });
      const data = await response.json();
      setMessages((list) => [...list.slice(0, -1), {
        question: asked, answer: data.answer, error: data.error,
        tools: data.tools?.map((t: { name: string }) => TOOL_LABELS[t.name] ?? t.name), untraced: data.untraced,
      }]);
    } catch {
      setMessages((list) => [...list.slice(0, -1), { question: asked, error: "Could not reach the assistant." }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button className="pitwall-button primary" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Close" : "Ask Pit Wall"}
      </button>
      {open && (
        <div className="pitwall-panel card" role="dialog" aria-label="Pit Wall assistant">
          <h3>Pit Wall assistant</h3>
          <p className="muted small" style={{ marginBottom: 10 }}>
            Answers come only from this site&apos;s model and are estimates, not facts.
          </p>
          {available === false ? (
            <p className="note small">The Pit Wall assistant is not available on this copy of the site.</p>
          ) : (
            <>
              <div className="pitwall-messages">
                {messages.length === 0 && (
                  <div className="stack" style={{ gap: 6 }}>
                    {EXAMPLES.map((example) => (
                      <button key={example} className="chip" style={{ textAlign: "left" }} onClick={() => ask(example)}>
                        {example}
                      </button>
                    ))}
                  </div>
                )}
                {messages.map((m, i) => (
                  <div key={i} className="pitwall-turn">
                    <p className="pitwall-question">{m.question}</p>
                    {m.answer && <p className="pitwall-answer">{m.answer}</p>}
                    {m.error && <p className="pitwall-answer muted">{m.error}</p>}
                    {!m.answer && !m.error && <p className="pitwall-answer muted">Checking the model...</p>}
                    {m.tools && m.tools.length > 0 && (
                      <p className="muted small">Used: {[...new Set(m.tools)].join(", ")}</p>
                    )}
                    {m.untraced && m.untraced.length > 0 && (
                      <p className="small" style={{ color: "var(--bad)" }}>
                        Could not verify these numbers against the model: {m.untraced.join(", ")}. Treat them with care.
                      </p>
                    )}
                  </div>
                ))}
                <div ref={bottom} />
              </div>
              <form className="row" onSubmit={(event) => { event.preventDefault(); ask(question); }}>
                <input value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={500}
                  placeholder="Ask about any driver..." aria-label="Your question" style={{ flex: 1 }} disabled={busy} />
                <button type="submit" disabled={busy || !question.trim()}>Ask</button>
              </form>
            </>
          )}
        </div>
      )}
    </>
  );
}

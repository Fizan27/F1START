// lib/rateLimit.ts
//
// What it does: stops a public link from running up an API bill. Every
//   question costs real money, so each visitor (by IP address) gets a few
//   questions per ten minutes and per day, and the whole site gets a daily cap.
//   The counts live in the server's memory. Vercel may run several copies of
//   the server and restarts them now and then, so this is a brake, not a
//   lock: the real backstop is the monthly spend limit you set in the
//   Anthropic Console (see the README).
// What it reads: optional environment variables PIT_WALL_PER_10_MIN,
//   PIT_WALL_PER_DAY and PIT_WALL_SITE_PER_DAY.
// What it produces: a yes/no answer and, when no, how long to wait.
// Which files use it: app/api/chat/route.ts and the tests.

const TEN_MINUTES = 10 * 60 * 1000;
const DAY = 24 * 60 * 60 * 1000;

export type Limits = { perTenMinutes: number; perDay: number; sitePerDay: number };

export function limitsFromEnvironment(env: Record<string, string | undefined> = process.env): Limits {
  const number = (name: string, fallback: number) => {
    const value = Number(env[name]);
    return Number.isFinite(value) && value > 0 ? value : fallback;
  };
  return {
    perTenMinutes: number("PIT_WALL_PER_10_MIN", 5),
    perDay: number("PIT_WALL_PER_DAY", 20),
    sitePerDay: number("PIT_WALL_SITE_PER_DAY", 100),
  };
}

export class RateLimiter {
  private visits = new Map<string, number[]>(); // visitor -> times of their questions
  private site: number[] = [];

  constructor(private limits: Limits) {}

  /** Record a question if it is allowed. Returns null if allowed, otherwise the reason. */
  check(visitor: string, now = Date.now()): string | null {
    this.site = this.site.filter((time) => now - time < DAY);
    const times = (this.visits.get(visitor) ?? []).filter((time) => now - time < DAY);
    if (this.site.length >= this.limits.sitePerDay) {
      return "The assistant has answered as many questions as it can today. Please come back tomorrow.";
    }
    if (times.length >= this.limits.perDay) return "You've reached today's question limit. Please come back tomorrow.";
    if (times.filter((time) => now - time < TEN_MINUTES).length >= this.limits.perTenMinutes) {
      return "That's a lot of questions in a short time. Please wait a few minutes.";
    }
    times.push(now);
    this.visits.set(visitor, times);
    this.site.push(now);
    return null;
  }
}

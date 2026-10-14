// tests/rateLimit.test.ts
//
// What it does: checks the rate limiter that stops a public link running up a bill.
// Which files use it: none.  Run:  cd web ; npm test

import { describe, expect, it } from "vitest";

import { limitsFromEnvironment, RateLimiter } from "@/lib/rateLimit";

const MINUTE = 60 * 1000;

describe("RateLimiter", () => {
  it("allows a few questions, then asks a visitor to wait, then allows again later", () => {
    const limiter = new RateLimiter({ perTenMinutes: 2, perDay: 10, sitePerDay: 100 });
    expect(limiter.check("a", 0)).toBeNull();
    expect(limiter.check("a", 1 * MINUTE)).toBeNull();
    expect(limiter.check("a", 2 * MINUTE)).toMatch(/wait/);
    expect(limiter.check("b", 2 * MINUTE)).toBeNull(); // other visitors are not affected
    expect(limiter.check("a", 11 * MINUTE)).toBeNull();
  });

  it("stops a visitor at the daily limit", () => {
    const limiter = new RateLimiter({ perTenMinutes: 100, perDay: 2, sitePerDay: 100 });
    limiter.check("a", 0);
    limiter.check("a", MINUTE);
    expect(limiter.check("a", 2 * MINUTE)).toMatch(/today/);
  });

  it("stops everyone when the site's daily cap is reached", () => {
    const limiter = new RateLimiter({ perTenMinutes: 100, perDay: 100, sitePerDay: 2 });
    limiter.check("a", 0);
    limiter.check("b", 0);
    expect(limiter.check("c", 0)).toMatch(/come back tomorrow/);
  });

  it("reads limits from environment variables, ignoring nonsense", () => {
    expect(limitsFromEnvironment({ PIT_WALL_PER_10_MIN: "3", PIT_WALL_PER_DAY: "banana" }))
      .toEqual({ perTenMinutes: 3, perDay: 20, sitePerDay: 100 });
  });
});

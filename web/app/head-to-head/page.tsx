// app/head-to-head/page.tsx
//
// What it does: the head to head page. Pick any two drivers and see the
//   chance each beats the other in identical cars, the uncertainty range of
//   that chance, and a plain English explanation built from the numbers.
// What it reads: drivers.json, h2h.json and details.json from public/data.
// Which files use it: none (it is a page).

"use client";

import Link from "next/link";
import { useState } from "react";

import DriverPicker from "@/components/DriverPicker";
import { type Driver, type DriverDetail, headToHead, type HeadToHeads, signed, useData } from "@/lib/data";

const TYPICAL_LAP_SECONDS = 90;

export default function HeadToHeadPage() {
  const drivers = useData<Driver[]>("drivers.json");
  const h2h = useData<HeadToHeads>("h2h.json");
  const details = useData<Record<string, DriverDetail>>("details.json");
  const [ids, setIds] = useState<[string, string]>(["senna", "max_verstappen"]);

  if (!drivers || !h2h) return <p className="muted">Loading...</p>;

  // Only drivers with enough races to have a meaningful rating are offered.
  const eligible = drivers.filter((driver) => h2h.drivers.includes(driver.id));
  const a = drivers.find((driver) => driver.id === ids[0])!;
  const b = drivers.find((driver) => driver.id === ids[1])!;
  const chance = headToHead(h2h, a.id, b.id);

  return (
    <>
      <h1>Head to head</h1>
      <p className="lead">
        Two drivers, each at their peak, in identical cars. How often does each finish ahead? Choose
        from the {eligible.length} drivers with at least 30 race starts.
      </p>

      <div className="grid-2" style={{ marginBottom: 16 }}>
        <DriverPicker drivers={eligible.filter((d) => d.id !== b.id)} onPick={(d) => setIds([d.id, b.id])} placeholder={`Change ${a.name}...`} />
        <DriverPicker drivers={eligible.filter((d) => d.id !== a.id)} onPick={(d) => setIds([a.id, d.id])} placeholder={`Change ${b.name}...`} />
      </div>

      {chance && (
        <div className="card">
          <div className="versus">
            <Side driver={a} percent={chance.best} colour="var(--series-1)" />
            <div className="muted">vs</div>
            <Side driver={b} percent={100 - chance.best} colour="var(--series-2)" />
          </div>
          <div className="split-bar" role="img" aria-label={`${a.name} ${chance.best}%, ${b.name} ${100 - chance.best}%`}>
            <div style={{ width: `${chance.best}%` }} />
            <div style={{ width: `${100 - chance.best}%` }} />
          </div>
          <p className="muted small" style={{ textAlign: "center" }}>
            Plausible range for {a.name}: <b className="num">{chance.low}% to {chance.high}%</b>
          </p>
          <Explanation a={a} b={b} chance={chance} details={details} />
        </div>
      )}
    </>
  );
}

function Side({ driver, percent, colour }: { driver: Driver; percent: number; colour: string }) {
  return (
    <div>
      <Link href={`/drivers/?id=${driver.id}`} style={{ fontWeight: 700, fontSize: 18 }}>
        {driver.name}
      </Link>
      <div className="muted small num">
        peak {driver.peak[0]}–{driver.peak[1]}
      </div>
      <div className="pct num">
        {percent}
        <span style={{ fontSize: "0.45em" }}>%</span>
      </div>
      <i style={{ display: "inline-block", width: 40, height: 4, borderRadius: 2, background: colour }} />
    </div>
  );
}

type ExplanationProps = {
  a: Driver;
  b: Driver;
  chance: { best: number; low: number; high: number };
  details: Record<string, DriverDetail> | null;
};

/** A few sentences that say what the numbers mean, written from the numbers themselves. */
function Explanation({ a, b, chance, details }: ExplanationProps) {
  const [ahead, behind] = a.skill >= b.skill ? [a, b] : [b, a];
  const gap = ahead.skill - behind.skill;
  const seconds = (gap / 100) * TYPICAL_LAP_SECONDS;
  const width = chance.high - chance.low;
  const yearsApart = Math.abs((a.peak[0] + a.peak[1]) / 2 - (b.peak[0] + b.peak[1]) / 2);
  const record = details?.[a.id]?.mates.find((mate) => mate[0] === b.id);

  const verdict =
    chance.low > 50 || chance.high < 50
      ? `Even at the unfavourable end of the range, ${chance.low > 50 ? a.name : b.name} stays the favourite.`
      : "The range crosses 50%, so the model cannot say with confidence who was better.";

  // How wide the range is, judged only from the range itself, and then the
  // real reasons for that width. (An earlier version picked its words from
  // the years apart, and could call a range "fairly narrow" BECAUSE the
  // drivers were decades apart, which is backwards.)
  const widthWord = width >= 45 ? "wide" : width >= 25 ? "fairly wide" : "fairly narrow";
  const reasons: string[] = [];
  if (record) {
    reasons.push("they were teammates, which ties their two ratings together and narrows it");
  } else if (yearsApart > 15) {
    reasons.push(`their peaks are ${Math.round(yearsApart)} years apart, so they are compared only through a long chain of teammates, and every link adds doubt`);
  } else {
    reasons.push("they never shared a car, so they are compared through drivers who were teammates of both, or of their teammates");
  }
  for (const driver of [a, b]) {
    const plusMinus = 1.645 * driver.sd;
    if (plusMinus >= 0.3) {
      reasons.push(`${driver.name}'s own rating is uncertain (give or take ${plusMinus.toFixed(2)}%), because of ${
        driver.starts < 80 ? "a short career" : driver.peak[0] < 1980 ? "an older era with fewer measured teammates" : "teammates who were themselves hard to rate"}`);
    }
  }

  return (
    <div className="prose" style={{ marginTop: 18 }}>
      <p>
        The model rates {ahead.name} at {signed(ahead.skill)}% and {behind.name} at {signed(behind.skill)}%
        (percent of lap time quicker than an average newcomer of today). That gap of {gap.toFixed(2)}% is
        about <b>{seconds.toFixed(2)} seconds a lap</b> on a 90 second lap. Small gaps still decide races:
        over a race distance it adds up, but one bad weekend can outweigh it.
      </p>
      <p>
        {verdict} The range ({chance.low}% to {chance.high}%) is {widthWord}. What sets its width: {reasons.join("; ")}.
      </p>
      {record && (
        <p>
          They really were teammates ({record[1] === record[2] ? record[1] : `${record[1]} to ${record[2]}`}). In
          races where neither car broke down, {a.name} finished ahead in <b>{record[3]} of {record[4]}</b>. In
          qualifying {a.name} was ahead in <b>{record[5]} of {record[6]}</b>.
        </p>
      )}
    </div>
  );
}

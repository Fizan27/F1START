# compare_ratings.py
#
# What it does: compares the ratings before and after the second round of
#   work (posterior sampling, ranks among drivers with 30+ starts): the top 20
#   side by side with their ranges, and every driver whose rank moved by more
#   than 10 places, with the reason. Also which seasons' equal car champion
#   changed with the reworked simulator.
# What it reads: data/drivers_before.json (a copy of the old
#   web/public/data/drivers.json, saved before the change) and the current
#   one, and the same for championships.json (data/championships_before.json).
#   web/public/data/drivers.json.
# What it produces: tables printed to the screen (quoted in the README and
#   DECISIONS.md 32).
# Which files use it: none.
#
# Run:  .venv\Scripts\python.exe compare_ratings.py

import json
from pathlib import Path

OLD_FILE = Path("data/drivers_before.json")
NEW_FILE = Path("web/public/data/drivers.json")
OLD_SEASONS = Path("data/championships_before.json")
NEW_SEASONS = Path("web/public/data/championships.json")
TOP = 20
BIG_MOVE = 10


def load(file: Path) -> dict:
    return {driver["id"]: driver for driver in json.loads(file.read_text(encoding="utf-8"))}


def skill_text(driver: dict) -> str:
    low, high = driver["skill"] - 1.645 * driver["sd"], driver["skill"] + 1.645 * driver["sd"]
    return f"{driver['skill']:+.2f} ({low:+.2f} to {high:+.2f})"


def rank_text(driver: dict) -> str:
    return "not ranked" if driver["rank"] is None else f"{driver['rank']} ({driver['rankLow']} to {driver['rankHigh']})"


def compare_championships():
    """Which seasons' equal car champion changed with the reworked simulator, and title chances."""
    before = {s["year"]: s for s in json.loads(OLD_SEASONS.read_text(encoding="utf-8"))}
    after = {s["year"]: s for s in json.loads(NEW_SEASONS.read_text(encoding="utf-8"))}
    changed, chances = [], []
    for year, season in after.items():
        old_top, new_top = before[year]["table"][0], season["table"][0]
        chances.append((old_top[2], new_top[2]))
        if old_top[0] != new_top[0]:
            changed.append(f"{year}: {old_top[0]} ({old_top[2]}%) -> {new_top[0]} ({new_top[2]}%)")
    old_mean = sum(c[0] for c in chances) / len(chances)
    new_mean = sum(c[1] for c in chances) / len(chances)
    print(f"\nEqual car championships: the favourite changed in {len(changed)} of {len(after)} seasons.")
    print(f"Average title chance of each season's favourite: {old_mean:.1f}% before, {new_mean:.1f}% after.")
    for line in changed:
        print("  " + line)


def why_moved(new: dict, skill_changed: bool) -> str:
    """The reason a driver's rank moved, from what changed between the two files."""
    if new["rank"] is None:
        return "fewer than 30 starts: no longer ranked"
    if skill_changed:
        return "rating changed"
    return "drivers with fewer than 30 starts above them are no longer ranked"


def main():
    old, new = load(OLD_FILE), load(NEW_FILE)
    top = sorted((d for d in new.values() if d["rank"] is not None), key=lambda d: d["rank"])[:TOP]
    print(f"{'#':>3} {'Driver':<24} {'Before: skill (90% range)':<28} {'rank (could rank)':<22} "
          f"{'After: skill (90% range)':<28} rank (could rank)")
    for driver in top:
        before = old[driver["id"]]
        print(f"{driver['rank']:>3} {driver['name']:<24} {skill_text(before):<28} {rank_text(before):<22} "
              f"{skill_text(driver):<28} {rank_text(driver)}")

    moved = []
    for key, after in new.items():
        before = old[key]
        if after["rank"] is None or before["rank"] is None:
            continue
        if abs(after["rank"] - before["rank"]) > BIG_MOVE:
            skill_changed = abs(after["skill"] - before["skill"]) > 0.005
            moved.append((after["rank"], after["name"], before["rank"], why_moved(after, skill_changed)))
    unranked = sum(1 for key, after in new.items() if after["rank"] is None)
    print(f"\n{len(moved)} ranked drivers moved more than {BIG_MOVE} places; {unranked} drivers with fewer than "
          "30 starts are no longer ranked at all.")
    print(f"The highest ranked of them (the highest is {min(moved)[0] if moved else '-'}):")
    for rank, name, before_rank, reason in sorted(moved)[:8]:
        print(f"  {name:<24} {before_rank:>4} -> {rank:<4} {reason}")
    changed = [d["name"] for k, d in new.items() if abs(d["skill"] - old[k]["skill"]) > 0.005]
    print(f"\nDrivers whose peak skill itself changed by more than 0.005: {len(changed)}")
    widened = sorted(((new[k]["sd"] / old[k]["sd"], new[k]["name"]) for k in new), reverse=True)
    middle = widened[len(widened) // 2][0]
    print(f"Peak uncertainty after / before: median {middle:.2f}, largest {widened[0][0]:.2f} ({widened[0][1]})")
    compare_championships()


if __name__ == "__main__":
    main()

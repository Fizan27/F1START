# export_web.py
#
# What it does: runs the simulations on the GPU and writes everything the
#   website needs as small JSON files. Nothing on the website is typed in by
#   hand: every number it shows comes from here.
#     1. Rank ranges: thousands of plausible versions of the ratings are
#        drawn from their uncertainty, and each driver's rank is noted in
#        every one. "Ranked 3rd to 14th" is the honest form of "ranked 6th".
#     2. Head to heads: the chance each driver beats each other driver in
#        identical cars, with a range.
#     3. Equal car championships: every real season replayed with its real
#        drivers and calendar, but everyone in the same car.
#     4. Overtaking difficulty per circuit, measured from real races.
# What it reads: the five files made by fit_ratings.py, data/results.parquet,
#   data/raw/*_laps.parquet and web/public/data/circuits/index.json.
# What it produces, in web/public/data/:
#     drivers.json        every driver: rating, range, consistency, rank
#     details.json        career curve and teammate records per driver
#     h2h.json            head to head chances for drivers with 30+ starts
#     championships.json  the equal car championship of every season
#     settings.json       the simulation's numbers, so the browser runs the
#                         same race as simulate.py, plus circuit difficulty
# Which files use it: the website (Phase 4).
#
# Run:  .venv\Scripts\python.exe export_web.py     (about 2 minutes)

import json
from pathlib import Path

import polars as pl
import torch

import simulate

DATA = Path("data")
RAW_FOLDER = DATA / "raw"
WEB = Path("web/public/data")

PLAUSIBLE_VERSIONS = 4000  # how many versions of the ratings are drawn
MIN_STARTS_FOR_H2H = 30  # head to heads are precomputed for these drivers
RACES_PER_SEASON_RACE = 1000  # simulations of each race of each season
TYPICAL_LAP_SECONDS = 90.0
TYPICAL_RACE_LAPS = 60
MODERN_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
RACE = ["Year", "Round"]


def write_json(name: str, content) -> None:
    file = WEB / name
    # separators without spaces keeps the files as small as possible
    file.write_text(json.dumps(content, separators=(",", ":")), encoding="utf-8")
    print(f"{name:<20} {file.stat().st_size / 1024:6.0f} KB")


# --- 1. Rank ranges ----------------------------------------------------------

def plausible_ratings(ratings: pl.DataFrame, saved: dict, generator) -> torch.Tensor:
    """Draw many plausible versions of every driver's peak rating at once.

    Returns versions x drivers. The draws respect the covariance: if two
    drivers were teammates, a version that rates one higher tends to rate the
    other higher too, so their GAP stays as certain as the data makes it.
    """
    mean = torch.tensor(ratings["PeakSkill"].to_list(), device=simulate.DEVICE, dtype=torch.float64)
    covariance = saved["covariance"].to(simulate.DEVICE, torch.float64)
    # A tiny amount is added to the diagonal so rounding errors cannot make
    # the matrix invalid for the square-root step (Cholesky) below.
    covariance = covariance + 1e-6 * torch.eye(len(mean), device=simulate.DEVICE, dtype=torch.float64)
    root = torch.linalg.cholesky(covariance)
    noise = torch.randn(PLAUSIBLE_VERSIONS, len(mean), device=simulate.DEVICE,
                        dtype=torch.float64, generator=generator)
    return (mean + noise @ root.T).float()


def rank_ranges(versions: torch.Tensor) -> tuple[list, list]:
    """Each driver's rank in the 5th and 95th percentile of the plausible versions."""
    ranks = (-versions).argsort(dim=1).argsort(dim=1).float() + 1
    low = ranks.quantile(0.05, dim=0).round().int().tolist()
    high = ranks.quantile(0.95, dim=0).round().int().tolist()
    return low, high


# --- 2. Head to heads --------------------------------------------------------

def head_to_heads(versions: torch.Tensor, race_slope: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The chance each driver beats each other in identical cars: best guess, low, high.

    In every plausible version of the ratings the chance is the model's
    S-curve of the skill gap (simulate.py is calibrated to the same curve).
    The best guess is the average over all versions. The range is where the
    middle 90% of versions fall: wide when the ratings are uncertain.
    """
    gap = versions[:, :, None] - versions[:, None, :]  # versions x drivers x drivers
    chance = torch.sigmoid(race_slope * gap)
    ordered = chance.sort(dim=0).values
    count = len(versions)
    return chance.mean(dim=0), ordered[int(count * 0.05)], ordered[int(count * 0.95)]


def upper_triangle(table: torch.Tensor) -> list[int]:
    """The chances for each pair once (A against B, not also B against A), in whole percent."""
    size = len(table)
    rows, columns = torch.triu_indices(size, size, offset=1)
    return (table[rows, columns] * 100).round().int().tolist()


# --- 3. Equal car championships ---------------------------------------------

def season_field(year: int, results: pl.DataFrame, seasons: pl.DataFrame, saved: dict):
    """The drivers of one season with their skill THAT season, and who started each race."""
    starters = results.filter((pl.col("Year") == year) & (pl.col("Outcome") != "not_started")
                              & ~pl.col("IsIndy500"))
    drivers = starters["DriverId"].unique(maintain_order=True).to_list()
    known = {driver: (skill, sd) for driver, skill, sd in
             seasons.filter(pl.col("Year") == year).select("DriverId", "Skill", "SkillSd").iter_rows()}
    # A driver who never had a teammate has no rating: assume an average
    # newcomer of that era, give or take the whole spread of drivers.
    unknown = (saved["era_level"][year // 10 * 10], saved["skill_spread"])
    skill = torch.tensor([known.get(driver, unknown) for driver in drivers], device=simulate.DEVICE)
    races = [[drivers.index(driver) for driver in race["DriverId"]]
             for _, race in starters.group_by("Round", maintain_order=True)]
    return drivers, skill, races


def simulate_season(skill: torch.Tensor, errors: torch.Tensor, races: list, generator) -> torch.Tensor:
    """Simulate every race of a season many times. Returns points, simulations x drivers."""
    simulations = RACES_PER_SEASON_RACE
    # One plausible version of each driver's skill per simulated season.
    version = skill[:, 0] + skill[:, 1] * torch.randn(
        simulations, len(skill), device=simulate.DEVICE, generator=generator)
    points_table = torch.tensor(MODERN_POINTS + [0] * 40, device=simulate.DEVICE, dtype=torch.float32)
    points = torch.zeros(simulations, len(skill), device=simulate.DEVICE)
    for starters in races:
        cars = torch.tensor(starters, device=simulate.DEVICE)
        place, crashed = simulate.simulate_races(
            version[:, cars], errors[cars], TYPICAL_LAP_SECONDS, TYPICAL_RACE_LAPS,
            simulate.pass_margin(0.5), generator)
        points[:, cars] += points_table[place - 1] * ~crashed
    return points


def championship(year, results, seasons, consistency: dict, saved, generator) -> dict:
    """One season's equal car championship next to what really happened."""
    drivers, skill, races = season_field(year, results, seasons, saved)
    scores = torch.tensor([consistency.get(driver, 50.0) for driver in drivers], device=simulate.DEVICE)
    points = simulate_season(skill, simulate.error_factor(scores), races, generator)
    champion = points.argmax(dim=1)
    title_share = torch.bincount(champion, minlength=len(drivers)).float() / len(points)
    real = dict(results.filter(pl.col("Year") == year).group_by("DriverId")
                .agg(pl.col("ModernPoints").sum()).iter_rows())
    table = [[driver, round(points[:, i].mean().item()), round(title_share[i].item() * 100, 1), real[driver]]
             for i, driver in enumerate(drivers)]
    table.sort(key=lambda row: -row[1])
    return {"year": year, "races": len(races), "table": table[:12]}


# --- 4. Overtaking difficulty -------------------------------------------------

def passes_per_circuit(results: pl.DataFrame) -> pl.DataFrame:
    """How often a car gains a place on track per lap, at each circuit (2018 onwards).

    Only cars that did not pit on this lap or the one before are compared, and
    only with each other, so a place gained because a rival pitted is not
    counted as an overtake.
    """
    columns = ["Year", "Round", "Driver", "LapNumber", "Position", "PitInTime", "PitOutTime", "TrackStatus"]
    laps = pl.concat([pl.read_parquet(file).select(columns) for file in sorted(RAW_FOLDER.glob("*_laps.parquet"))],
                     how="vertical_relaxed")
    racing = laps.filter(pl.col("PitInTime").is_null() & pl.col("PitOutTime").is_null()
                         & (pl.col("TrackStatus") == "1") & pl.col("Position").is_not_null()
                         & (pl.col("LapNumber") > 2)).with_columns(pl.col("LapNumber").cast(pl.Int64))
    before = racing.select(*RACE, "Driver", (pl.col("LapNumber") + 1).alias("LapNumber"),
                           pl.col("Position").alias("PositionBefore"))
    both = racing.join(before, on=RACE + ["Driver", "LapNumber"])
    same_lap = RACE + ["LapNumber"]
    gained = pl.col("Position").rank().over(same_lap) < pl.col("PositionBefore").rank().over(same_lap)
    per_race = both.group_by(RACE).agg(gained.mean().alias("PassRate"))
    circuits = results.select(*RACE, "CircuitId").unique()
    return per_race.join(circuits, on=RACE).group_by("CircuitId").agg(
        pl.col("PassRate").mean(), pl.len().alias("Races"))


def overtaking_difficulty(results: pl.DataFrame, circuit_ids: list[str]) -> dict:
    """A difficulty from 0 (easiest to pass) to 1 (hardest) for each circuit with an outline."""
    passes = passes_per_circuit(results).filter(pl.col("CircuitId").is_in(circuit_ids)).sort("PassRate")
    # Rank based: the circuit with the fewest passes gets 1, the most gets 0.
    step = 1 / (passes.height - 1)
    difficulty = {circuit: round(1 - place * step, 2) for place, circuit in enumerate(passes["CircuitId"])}
    # A circuit with an outline but no usable laps is treated as middling.
    return {circuit: difficulty.get(circuit, 0.5) for circuit in circuit_ids}


# --- Putting it together ------------------------------------------------------

def driver_rows(ratings: pl.DataFrame, rank_low: list, rank_high: list) -> list[dict]:
    """The compact list of every driver for the ranking page and the driver picker."""
    order = ratings.with_row_index("row").sort("PeakSkill", descending=True)
    rows = []
    for rank, driver in enumerate(order.iter_rows(named=True), start=1):
        rows.append({
            "id": driver["DriverId"], "name": driver["DriverName"], "rank": rank,
            "rankLow": rank_low[driver["row"]], "rankHigh": rank_high[driver["row"]],
            "skill": round(driver["PeakSkill"], 3), "sd": round(driver["PeakSd"], 3),
            "peak": [driver["PeakFrom"], driver["PeakTo"]],
            "years": [driver["FirstYear"], driver["LastYear"]],
            "starts": driver["Starts"], "wins": driver["Wins"], "mates": driver["Teammates"],
            "consistency": round(driver["Consistency"]),
        })
    return rows


def detail_rows(ratings, seasons: pl.DataFrame, records: pl.DataFrame) -> dict:
    """Per driver: the career curve, teams, crash record, and the eight most-raced teammates."""
    curves, mates = {}, {}
    for driver, year, skill, sd in seasons.sort("DriverId", "Year").iter_rows():
        curves.setdefault(driver, []).append([year, round(skill, 3), round(sd, 3)])
    for row in records.iter_rows(named=True):
        if len(mates.setdefault(row["DriverId"], [])) < 8:
            mates[row["DriverId"]].append([
                row["Teammate"], row["FromYear"], row["ToYear"], row["RaceWins"], row["RacesCompared"],
                row["QualiWins"], row["QualisCompared"]])
    details = {}
    for driver in ratings.iter_rows(named=True):
        details[driver["DriverId"]] = {
            "curve": curves[driver["DriverId"]], "mates": mates.get(driver["DriverId"], []),
            "teams": driver["Teams"],
            "crashRatio": None if driver["CrashRatio"] is None else round(driver["CrashRatio"], 2),
            "lapSpreadRatio": None if driver["LapSpreadRatio"] is None else round(driver["LapSpreadRatio"], 2),
        }
    return details


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=simulate.DEVICE).manual_seed(2026)
    ratings = pl.read_parquet(DATA / "ratings.parquet")
    seasons = pl.read_parquet(DATA / "skill_by_season.parquet")
    records = pl.read_parquet(DATA / "teammate_records.parquet")
    results = pl.read_parquet(DATA / "results.parquet")
    saved = torch.load(DATA / "peak_covariance.pt")
    assert saved["drivers"] == ratings["DriverId"].to_list(), "run fit_ratings.py again"
    print(f"Simulating on {simulate.DEVICE}. Files written to {WEB}:")

    versions = plausible_ratings(ratings, saved, generator)
    rank_low, rank_high = rank_ranges(versions)
    write_json("drivers.json", driver_rows(ratings, rank_low, rank_high))
    write_json("details.json", detail_rows(ratings, seasons, records))

    regulars = [row for row, starts in enumerate(ratings["Starts"]) if starts >= MIN_STARTS_FOR_H2H]
    best, low, high = head_to_heads(versions[:, regulars], saved["scales"]["race_slope"])
    write_json("h2h.json", {
        "drivers": [ratings["DriverId"][row] for row in regulars],
        "best": upper_triangle(best), "low": upper_triangle(low), "high": upper_triangle(high)})

    consistency = dict(ratings.select("DriverId", "Consistency").iter_rows())
    years = sorted(results["Year"].unique().to_list())
    write_json("championships.json", [
        championship(year, results, seasons, consistency, saved, generator) for year in years])

    circuits = json.loads((WEB / "circuits" / "index.json").read_text(encoding="utf-8"))
    write_json("settings.json", {
        "sim": simulate.SETTINGS, "raceSlope": round(saved["scales"]["race_slope"], 3),
        "difficulty": overtaking_difficulty(results, [circuit["id"] for circuit in circuits]),
    })


if __name__ == "__main__":
    main()

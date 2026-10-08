# download_circuits.py
#
# What it does: draws the outline of every circuit raced since 2018, for the
#   website's race animation. For each circuit it takes the car positions
#   recorded during the fastest lap of the most recent race there, and turns
#   them into a small, clean list of points.
# What it reads: data/results.parquet (made by clean_history.py) to know which
#   circuits exist and when they were last raced, and FastF1 (internet needed)
#   for the car positions.
# What it produces, in web/public/data/circuits/:
#     silverstone.json, monza.json, ...  one small file per circuit
#     index.json                         the list of circuits, for the menu
# Which files use it: the website (Phase 4) reads these files directly.
#
# Run:  .venv\Scripts\python.exe download_circuits.py
# Safe to stop and run again: circuits already saved are skipped.
# Expect about 10 minutes the first time, and about 2 GB more in the cache.

import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import polars as pl

from download_laps import CACHE_FOLDER, FIRST_SEASON, patiently

RESULTS_FILE = Path("data/results.parquet")
OUTPUT_FOLDER = Path("web/public/data/circuits")

# Every outline is stored as this many points, the same distance apart. The
# website can then place a car that is 37% of the way round a lap at point
# number 0.37 x 240, with no further maths.
POINTS_PER_CIRCUIT = 240
BOX_SIZE = 1000  # outlines are scaled to fit a 1000 x 1000 square
MIN_DIFFERENT_POSITIONS = 150  # fewer than this in a lap means a frozen feed


def races_per_circuit(results: pl.DataFrame) -> dict[str, list[dict]]:
    """Every race since 2018 at each circuit, newest first.

    The newest is tried first because layouts change (Abu Dhabi and Melbourne
    were both reshaped), and the newest layout is the one people know. Older
    races are the fallback when a race has no position data.
    """
    races = results.filter(pl.col("Year") >= FIRST_SEASON).group_by("Year", "Round").agg(
        pl.col("CircuitId", "CircuitName", "Country").first(),
        pl.col("Laps").max().alias("RaceLaps"),
    )
    by_circuit = {}
    for race in races.sort("Year", "Round", descending=True).to_dicts():
        by_circuit.setdefault(race["CircuitId"], []).append(race)
    return dict(sorted(by_circuit.items()))


def fastest_lap_positions(year: int, round_number: int) -> tuple[np.ndarray, float, float]:
    """Car positions over the race's fastest lap.

    Returns (points as rows of x and y, lap time in seconds, map rotation in
    degrees). The rotation turns the map the way it is usually drawn on TV.
    """
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    lap_seconds = session.laps.pick_fastest()["LapTime"].total_seconds()
    # Position data is sometimes missing for one car, so try the quickest few
    # laps in turn until one has it.
    for _, lap in session.laps.pick_quicklaps().sort_values("LapTime").head(20).iterrows():
        positions = lap.get_pos_data()
        if "X" not in positions:
            continue
        points = positions[["X", "Y"]].to_numpy(dtype=float)
        if has_enough_detail(points):
            return points, lap_seconds, map_rotation(session)
    raise ValueError("no lap in this race has good position data")


def has_enough_detail(points: np.ndarray) -> bool:
    """False when the recording is too coarse to draw a smooth track.

    A good lap has about 300 different positions. In a few races the feed
    froze for seconds at a time, repeating the same position, which would
    draw the track as a rough polygon.
    """
    return len(np.unique(points, axis=0)) >= MIN_DIFFERENT_POSITIONS


def map_rotation(session) -> float:
    """Degrees to turn the map. 0 for new circuits FastF1 has no map details for."""
    try:
        return session.get_circuit_info().rotation
    except Exception:
        return 0.0


def resample_evenly(points: np.ndarray, count: int) -> np.ndarray:
    """Replace a closed loop of points with `count` points the same distance apart.

    The recorded points are evenly spaced in TIME (about four a second), so
    they bunch up in slow corners and spread out on straights. Evenly spaced
    in DISTANCE is what an animation needs.
    """
    loop = np.vstack([points, points[:1]])  # join the end back to the start
    step_lengths = np.hypot(*np.diff(loop, axis=0).T)
    distance = np.concatenate([[0.0], np.cumsum(step_lengths)])
    wanted = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.column_stack([
        np.interp(wanted, distance, loop[:, 0]),
        np.interp(wanted, distance, loop[:, 1]),
    ])


def rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Turn the whole outline around the origin by an angle."""
    angle = np.radians(degrees)
    turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    return points @ turn


def fit_to_box(points: np.ndarray, size: int) -> np.ndarray:
    """Scale an outline to fit a square, as whole numbers, keeping its shape.

    Both directions are scaled by the same amount so the track is not
    stretched. Y is flipped because on a screen y grows downwards.
    """
    x = points[:, 0] - points[:, 0].min()
    y = points[:, 1].max() - points[:, 1]
    scale = size / max(x.max(), y.max())
    return np.rint(np.column_stack([x, y]) * scale).astype(int)


def build_circuit(race: dict) -> dict:
    """Download and tidy one circuit's outline."""
    points, lap_seconds, rotation = patiently(fastest_lap_positions, race["Year"], race["Round"])
    outline = fit_to_box(rotate(resample_evenly(points, POINTS_PER_CIRCUIT), rotation), BOX_SIZE)
    return {
        "id": race["CircuitId"],
        "name": race["CircuitName"],
        "country": race["Country"],
        "year": race["Year"],  # the race the outline and lap time come from
        "raceLaps": race["RaceLaps"],
        "lapSeconds": round(lap_seconds, 3),
        "points": outline.tolist(),
    }


def write_index():
    """Save the list of circuits (everything except the points) for the website's menu."""
    circuits = []
    for file in sorted(OUTPUT_FOLDER.glob("*.json")):
        if file.name != "index.json":
            circuit = json.loads(file.read_text(encoding="utf-8"))
            circuits.append({key: value for key, value in circuit.items() if key != "points"})
    (OUTPUT_FOLDER / "index.json").write_text(json.dumps(circuits, indent=1), encoding="utf-8")
    return len(circuits)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_FOLDER))
    logging.getLogger("fastf1").setLevel(logging.CRITICAL)  # hide progress chatter

    for circuit_id, races in races_per_circuit(pl.read_parquet(RESULTS_FILE)).items():
        file = OUTPUT_FOLDER / f"{circuit_id}.json"
        if file.exists():
            continue
        for race in races:
            try:
                circuit = build_circuit(race)
            except Exception as error:
                # One bad race must not stop the rest: try the one before it.
                print(f"{circuit_id:<16} {race['Year']}  FAILED: {error}")
                continue
            # separators without spaces keeps the file as small as possible
            file.write_text(json.dumps(circuit, separators=(",", ":")), encoding="utf-8")
            print(f"{circuit_id:<16} {race['Year']}  lap {circuit['lapSeconds']:.1f}s")
            break

    print(f"\nDone. {write_index()} circuits in {OUTPUT_FOLDER}.")


if __name__ == "__main__":
    main()

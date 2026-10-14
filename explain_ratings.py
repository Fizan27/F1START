# explain_ratings.py
#
# What it does: works out WHY each driver has the rating they have, for the
#   "Why this rating" panel on the website.
#   1. Which teammate battles moved the rating most, up and down. For each
#      pair of teammates (Senna and Prost, 1988 to 1989, say) it estimates how
#      much the driver's peak rating would change if that pair's results were
#      removed. Battles the driver was not in count too: if a rival's old
#      teammate is rated higher, that changes the rival, and the change
#      travels along the chain.
#   2. The chain of teammates linking the driver to today's grid: the
#      shortest route through shared teammates, preferring teammates who
#      raced together for longer.
#
#   How (1) is done without refitting thousands of times: removing a set of
#   results nudges the best fit by roughly covariance x (how hard those
#   results were pulling). It is the same covariance the Laplace step makes,
#   and it is a standard approximation ("influence functions"). It is
#   accurate for small nudges and only roughly right for big ones.
#
# What it reads: the same tables as model.py, and data/ratings.parquet.
# What it produces: web/public/data/explain.json.
# Which files use it: web/app/drivers/page.tsx and the assistant's
#   explain_rating tool.
#
# Run:  .venv\Scripts\python.exe explain_ratings.py     (about a minute)

import heapq
import json
from pathlib import Path

import polars as pl
import torch

import fit_ratings
import model

OUTPUT_FILE = Path("web/public/data/explain.json")
BATTLES_SHOWN = 4  # per direction
SMALLEST_SHOWN = 0.005  # battles that move the rating less than this are left out


def pull_of_each_result(fit: model.Fit) -> dict:
    """For every comparison: how hard it pulls on the skill gap between the two drivers.

    In maths terms, the slope of that comparison's log likelihood with respect
    to the gap, at the best fit. Positive means "this result wants A rated
    higher relative to B".
    """
    data, scales = fit.data, fit.scales
    parts = model.split(data["unknowns"], data["layout"])
    pulls = {}
    for kind, rows in data["results"].items():
        gap = parts["skill"][rows["a"]] - parts["skill"][rows["b"]]
        if parts["wet"] is not None and rows["wet"] is not None:
            gap = gap + rows["wet"] * (parts["wet"][rows["driver_a"]] - parts["wet"][rows["driver_b"]])
        gap = gap.detach().requires_grad_(True)
        single = {"race_ahead": None, "quali_ahead": None}
        if kind in single:
            slope = scales["race_slope"] if kind == "race_ahead" else model.AHEAD_SLOPE_PER_NOISE / scales["quali_noise"]
            chance = -torch.nn.functional.binary_cross_entropy_with_logits(slope * gap, rows["value"], reduction="none")
        else:
            factor = scales["pace_factor"] if kind == "pace_gap" else 1.0
            noise = {"quali_gap": "quali_noise", "old_quali_gap": "old_quali_noise", "pace_gap": "pace_noise"}[kind]
            chance = model.heavy_tailed_log_chance(rows["value"], -factor * gap, scales[noise])
        (rows["weight"] * chance).sum().backward()
        pulls[kind] = gap.grad
    return pulls


def battle_effects(fit: model.Fit, weights: torch.Tensor) -> tuple[torch.Tensor, list]:
    """How much each teammate pairing moves each driver's peak rating: pairings x drivers.

    Returns the table and the pairings it refers to (DriverA, DriverB, first
    and last season, races). The number is the rating WITH the pairing minus
    the rating WITHOUT it, so positive means the battle pushed the rating up.
    """
    data = fit.data
    start, end = fit.layout["skill"]
    full = torch.zeros(weights.shape[0], fit.layout["total"], dtype=model.DTYPE, device=model.DEVICE)
    full[:, start:end] = weights[:, start:end]
    # Column j of `reach` = how each unknown is tied to driver j's peak.
    reach = fit.covariance @ full.T
    pulls = pull_of_each_result(fit)

    comparisons = data["comparisons"].with_row_index("row")
    pairings = comparisons.group_by("DriverA", "DriverB").agg(
        pl.col("Year").min().alias("From"), pl.col("Year").max().alias("To"),
        pl.col("Round").count().alias("Races")).sort("DriverA", "DriverB").with_row_index("group")
    group_of_row = comparisons.join(pairings.select("DriverA", "DriverB", "group"), on=["DriverA", "DriverB"])
    group_of_row = torch.tensor(group_of_row.sort("row")["group"].to_list(), device=model.DEVICE)

    effects = torch.zeros(pairings.height, weights.shape[0], dtype=model.DTYPE, device=model.DEVICE)
    wet_start = fit.layout["wet"][0]
    for kind, rows in data["results"].items():
        if rows["a"].numel() == 0:
            continue
        # Which comparison rows of this kind belong to which pairing.
        kind_rows = torch.tensor(comparisons.filter(pl.col(kind).is_not_null())["row"].to_list(), device=model.DEVICE)
        moved = reach[rows["a"]] - reach[rows["b"]]  # comparisons x drivers
        if fit.wet and rows["wet"] is not None:
            moved = moved + rows["wet"][:, None] * (reach[wet_start + rows["driver_a"]] - reach[wet_start + rows["driver_b"]])
        effects.index_add_(0, group_of_row[kind_rows], pulls[kind][:, None] * moved)
    return effects, pairings.drop("group").rows()


def chains_to_today(pairs: pl.DataFrame, latest: int) -> dict:
    """For every driver, the chain of teammates to someone on today's grid.

    Shortest in links first; among equally short chains, the one whose
    teammates raced together most (each link costs 1 + 1 / races together).
    Worked out once, backwards from today's drivers, for everyone at once.
    """
    together = pairs.group_by("DriverA", "DriverB").agg(
        pl.len().alias("races"), pl.col("Year").min().alias("from"), pl.col("Year").max().alias("to"))
    links = {}
    for a, b, races, first, last in together.iter_rows():
        links.setdefault(a, []).append((b, races, first, last))
        links.setdefault(b, []).append((a, races, first, last))
    today = set(pairs.filter(pl.col("Year") == latest).select(pl.concat_list("DriverA", "DriverB")).explode("DriverA")["DriverA"])
    cost = {driver: 0.0 for driver in today}
    step = {}  # driver -> (next driver towards today, races, first, last)
    queue = [(0.0, driver) for driver in today]
    heapq.heapify(queue)
    while queue:
        so_far, driver = heapq.heappop(queue)
        if so_far > cost[driver]:
            continue
        for mate, races, first, last in links.get(driver, []):
            new = so_far + 1 + 1 / races
            if new < cost.get(mate, float("inf")):
                cost[mate] = new
                step[mate] = (driver, races, first, last)
                heapq.heappush(queue, (new, mate))
    chains = {}
    for driver in links:
        chain, here = [], driver
        while here in step:
            nxt, races, first, last = step[here]
            chain.append([nxt, first, last, races])
            here = nxt
        chains[driver] = chain
    return chains


def main():
    pairs, pace = model.load_tables()
    ratings = pl.read_parquet("data/ratings.parquet")
    latest = pairs["Year"].max()
    fit = model.fit_model(pairs, pace, latest, wet_races=model.load_wet_races())
    _, _, weights = fit_ratings.peak_ratings(fit)
    drivers = list(dict.fromkeys(driver for driver, _ in fit.index))  # same order as the weights
    effects, pairings = battle_effects(fit, weights)
    chains = chains_to_today(pairs, latest)

    explained = {}
    for column, driver in enumerate(drivers):
        order = effects[:, column].argsort()

        def battles(positions):
            shown = []
            for position in positions.tolist():
                value = effects[position, column].item()
                if abs(value) >= SMALLEST_SHOWN:
                    a, b, first, last, races = pairings[position]
                    shown.append([a, b, first, last, races, round(value, 3)])
            return shown

        explained[driver] = {
            "up": battles(order.flip(0)[:BATTLES_SHOWN]),
            "down": battles(order[:BATTLES_SHOWN]),
            "chain": chains.get(driver, []),
        }
    OUTPUT_FILE.write_text(json.dumps(explained, separators=(",", ":")), encoding="utf-8")
    print(f"Saved explanations for {len(explained)} drivers to {OUTPUT_FILE} "
          f"({OUTPUT_FILE.stat().st_size / 1024:.0f} KB)")
    for driver in ["hamilton", "michael_schumacher", "senna", "frentzen"]:
        name = dict(ratings.select("DriverId", "DriverName").iter_rows()).get(driver, driver)
        print(f"\n{name}: up {explained[driver]['up'][:3]}\n  down {explained[driver]['down'][:3]}"
              f"\n  chain {[link[0] for link in explained[driver]['chain']]}")


if __name__ == "__main__":
    main()
